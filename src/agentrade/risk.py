"""Deterministic pre-trade risk checks.

The agent never talks to the broker directly: every order goes through RiskManager.check().
Rejections come back with plain-language reasons so the agent can resize or drop the idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RiskLimits
from .models import Account, Order, OrderRequest, OrderType, Position, RiskDecision, Side


@dataclass
class Exposure:
    """Positions and cash as they will be once currently open orders fill."""

    qty: dict[str, float]
    value: dict[str, float]
    cash: float

    @property
    def gross(self) -> float:
        return sum(abs(v) for v in self.value.values())

    @property
    def open_symbols(self) -> set[str]:
        return {s for s, q in self.qty.items() if abs(q) > 1e-9}


def projected_exposure(account: Account, positions: dict[str, Position],
                       open_orders: list[Order], prices: dict[str, float]) -> Exposure:
    qty = {s: p.qty for s, p in positions.items()}
    cash = account.cash
    for o in open_orders:
        price = o.request.limit_price or prices.get(o.symbol, 0.0)
        signed = o.remaining_qty if o.request.side == Side.BUY else -o.remaining_qty
        qty[o.symbol] = qty.get(o.symbol, 0.0) + signed
        cash -= signed * price
    value = {}
    for s, q in qty.items():
        price = prices.get(s) or (positions[s].market_price if s in positions else 0.0)
        value[s] = q * price
    return Exposure(qty=qty, value=value, cash=cash)


class RiskManager:
    def __init__(self, limits: RiskLimits, allowed_symbols: list[str] | None = None,
                 kill_switch_path: Path | None = None):
        self.limits = limits
        self.allowed_symbols = {s.upper() for s in allowed_symbols} if allowed_symbols else None
        self.kill_switch_path = kill_switch_path

    def kill_switch_active(self) -> bool:
        return bool(self.kill_switch_path and self.kill_switch_path.exists())

    def check(self, order: OrderRequest, price: float, account: Account,
              exposure: Exposure, orders_today: int) -> RiskDecision:
        L = self.limits
        reasons: list[str] = []

        if self.kill_switch_active():
            return RiskDecision(False, ["Kill switch is active: all trading is halted."])
        if order.qty <= 0:
            return RiskDecision(False, ["Quantity must be positive."])
        if price <= 0:
            return RiskDecision(False, [f"No valid price for {order.symbol}."])
        if self.allowed_symbols is not None and order.symbol not in self.allowed_symbols:
            reasons.append(f"{order.symbol} is not in the allowed watchlist.")
        if orders_today >= L.max_orders_per_day:
            reasons.append(f"Daily order limit reached ({L.max_orders_per_day}).")
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            reasons.append("Limit orders need a limit_price.")

        exec_price = order.limit_price if order.order_type == OrderType.LIMIT and order.limit_price else price
        notional = order.qty * exec_price
        current_qty = exposure.qty.get(order.symbol, 0.0)
        signed = order.qty if order.side == Side.BUY else -order.qty
        new_qty = current_qty + signed
        reduces_risk = abs(new_qty) < abs(current_qty) - 1e-9 and current_qty * new_qty >= 0

        if reduces_risk:
            # Closing or trimming is always allowed (unless halted), so the agent can always de-risk.
            return RiskDecision(not reasons, reasons)

        if new_qty < -1e-9 and not L.allow_short:
            reasons.append(
                f"Short selling is disabled: you hold {current_qty:g} {order.symbol}, can't sell {order.qty:g}."
            )
        if notional > L.max_order_notional:
            reasons.append(f"Order notional ${notional:,.0f} exceeds the ${L.max_order_notional:,.0f} per-order cap.")

        if account.day_pnl_pct <= -L.max_daily_loss_pct:
            reasons.append(
                f"Daily loss limit hit ({account.day_pnl_pct:.2%} vs -{L.max_daily_loss_pct:.0%}): "
                "only risk-reducing orders are allowed today."
            )

        equity = account.equity
        new_value = abs(new_qty) * exec_price
        if equity > 0 and new_value / equity > L.max_position_pct + 1e-9:
            max_qty = max(L.max_position_pct * equity / exec_price - abs(current_qty), 0)
            reasons.append(
                f"{order.symbol} would be {new_value / equity:.1%} of equity (max {L.max_position_pct:.0%}); "
                f"at most {max_qty:.2f} more shares fit."
            )

        new_gross = exposure.gross - abs(exposure.value.get(order.symbol, 0.0)) + new_value
        if equity > 0 and new_gross / equity > L.max_gross_exposure_pct + 1e-9:
            reasons.append(
                f"Gross exposure would be {new_gross / equity:.1%} of equity (max {L.max_gross_exposure_pct:.0%})."
            )

        opens_new = abs(current_qty) < 1e-9
        if opens_new and len(exposure.open_symbols) >= L.max_open_positions:
            reasons.append(f"Already at the max of {L.max_open_positions} open positions.")

        if order.side == Side.BUY and notional > exposure.cash + 1e-6:
            reasons.append(f"Insufficient cash: need ${notional:,.0f}, have ${exposure.cash:,.0f} after open orders.")

        if order.side == Side.BUY and L.require_stop_loss:
            if order.stop_loss_price is None:
                reasons.append("A stop_loss_price is required for every new or increased long position.")
            elif order.stop_loss_price >= exec_price:
                reasons.append(f"Stop ${order.stop_loss_price:.2f} must be below the entry price ${exec_price:.2f}.")
            elif 1 - order.stop_loss_price / exec_price > L.max_stop_distance_pct + 1e-9:
                reasons.append(
                    f"Stop is {1 - order.stop_loss_price / exec_price:.1%} below entry "
                    f"(max {L.max_stop_distance_pct:.0%})."
                )

        return RiskDecision(not reasons, reasons)
