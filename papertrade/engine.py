"""Motor común: controles de riesgo, cálculo de órdenes y simulación día a día.

Regla de tiempo (la misma en backtest y en paper trading simulado):
la decisión se toma con el cierre del día t y las órdenes se ejecutan a la
apertura del día t+1. Nunca se opera al mismo precio que generó la señal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Callable

import pandas as pd

from papertrade.config import BrokerConfig, Config, RiskConfig
from papertrade.data import MarketData
from papertrade.strategies import Decision, Strategy


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    close_position: bool = False  # vender toda la posición (evita restos por redondeo)


@dataclass
class Fill:
    date: str
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    slippage_cost: float


def apply_risk(weights: dict[str, float], risk: RiskConfig, universe: list[str]) -> dict[str, float]:
    """Sin cortos, tope por ticker y una reserva mínima de efectivo para costos."""
    clean = {}
    for symbol in universe:
        w = float(weights.get(symbol, 0.0) or 0.0)
        clean[symbol] = 0.0 if not math.isfinite(w) else min(max(w, 0.0), risk.max_weight)
    total = sum(clean.values())
    cap = 1.0 - risk.cash_buffer
    if total > cap:
        clean = {s: w * cap / total for s, w in clean.items()}
    return clean


def current_weights(positions: dict[str, float], prices, equity: float) -> dict[str, float]:
    if equity <= 0:
        return {s: 0.0 for s in positions}
    return {s: q * float(prices[s]) / equity for s, q in positions.items()}


def _round_down(qty: float, fractional: bool) -> float:
    return math.floor(qty * 1e6) / 1e6 if fractional else float(math.floor(qty + 1e-9))


def plan_orders(
    targets: dict[str, float],
    positions: dict[str, float],
    prices,
    equity: float,
    band: float,
    fractional: bool = False,
) -> list[Order]:
    """Órdenes para llevar el portafolio a `targets`. Ventas primero para liberar efectivo."""
    sells, buys = [], []
    held = {s for s, q in positions.items() if q > 0}
    for symbol in sorted(set(targets) | held):
        price = float(prices[symbol])
        qty_now = max(positions.get(symbol, 0.0), 0.0)
        target = targets.get(symbol, 0.0)
        if target <= 0:
            if qty_now > 0:
                sells.append(Order(symbol, "sell", qty_now, close_position=True))
            continue
        weight_now = qty_now * price / equity if equity > 0 else 0.0
        if qty_now > 0 and abs(target - weight_now) < band:
            continue
        delta = (target * equity - qty_now * price) / price
        qty = _round_down(abs(delta), fractional)
        if qty <= 0:
            continue
        (buys if delta > 0 else sells).append(Order(symbol, "buy" if delta > 0 else "sell", qty))
    return sells + buys


class SimBroker:
    """Broker simulado con comisiones y slippage. Nunca deja el efectivo en negativo."""

    def __init__(self, cfg: BrokerConfig):
        self.cfg = cfg
        self.cash = float(cfg.starting_cash)
        self.positions: dict[str, float] = {}

    def equity(self, prices) -> float:
        return self.cash + sum(q * float(prices[s]) for s, q in self.positions.items())

    def execute(self, order: Order, price: float, day: str) -> Fill | None:
        slip = self.cfg.slippage_bps / 10_000
        bps = self.cfg.commission_bps / 10_000
        fixed = self.cfg.commission_per_order
        held = self.positions.get(order.symbol, 0.0)

        if order.side == "sell":
            qty = min(order.qty, held)
            if qty <= 0:
                return None
            px = price * (1 - slip)
            gross = qty * px
            commission = fixed + gross * bps
            self.cash += gross - commission
            remaining = held - qty
            if remaining > 1e-9:
                self.positions[order.symbol] = remaining
            else:
                self.positions.pop(order.symbol, None)
        else:
            px = price * (1 + slip)
            affordable = (self.cash - fixed) / (px * (1 + bps))
            qty = min(order.qty, _round_down(max(affordable, 0.0), self.cfg.fractional))
            if qty <= 0:
                return None
            gross = qty * px
            commission = fixed + gross * bps
            self.cash -= gross + commission
            self.positions[order.symbol] = held + qty

        return Fill(day, order.symbol, order.side, qty, px, commission, qty * abs(px - price))


def is_due(day: pd.Timestamp, last_decision: pd.Timestamp | None, rebalance: str) -> bool:
    if last_decision is None:
        return True
    if day <= last_decision:
        return False
    if rebalance == "daily":
        return True
    if rebalance == "weekly":
        return day.isocalendar()[:2] != last_decision.isocalendar()[:2]
    return (day.year, day.month) != (last_decision.year, last_decision.month)


def decide_step(
    strategy: Strategy,
    cfg: Config,
    day: pd.Timestamp,
    data: MarketData,
    current: dict[str, float],
    drawdown: float,
    last_decision: pd.Timestamp | None,
) -> Decision | None:
    """Aplica el kill switch y el calendario de rebalanceo antes de consultar a la estrategia."""
    stop = cfg.risk.max_drawdown_stop
    if stop > 0 and drawdown >= stop:
        if not any(w > 0 for w in current.values()):
            return None
        notes = {"kill_switch": f"caída {drawdown:.1%} >= límite {stop:.1%}; se liquida todo"}
        return Decision({s: 0.0 for s in current}, notes)
    if not is_due(day, last_decision, cfg.rebalance):
        return None
    decision = strategy.decide(day, data.until(day), current)
    raw = {s: round(float(w), 6) for s, w in decision.weights.items()}
    decision.weights = apply_risk(decision.weights, cfg.risk, cfg.symbols)
    if raw != {s: round(w, 6) for s, w in decision.weights.items()}:
        decision.notes["pesos_antes_de_riesgo"] = raw
    return decision


DecideFn = Callable[[pd.Timestamp, MarketData, dict[str, float], float], Decision | None]


@dataclass
class SimResult:
    equity: pd.DataFrame  # columnas: equity, cash
    fills: list[Fill]
    decisions: list[tuple[pd.Timestamp, Decision]]
    positions: dict[str, float]
    cash: float
    pending: list[Order] = field(default_factory=list)

    def fills_frame(self) -> pd.DataFrame:
        cols = [f.name for f in fields(Fill)]
        return pd.DataFrame([asdict(f) for f in self.fills], columns=cols)


def simulate(cfg: Config, data: MarketData, start: pd.Timestamp, decide: DecideFn) -> SimResult:
    broker = SimBroker(cfg.broker)
    pending: list[Order] = []
    fills: list[Fill] = []
    decisions: list[tuple[pd.Timestamp, Decision]] = []
    rows = []
    peak = 0.0

    for day in data.dates[data.dates >= start]:
        label = day.strftime("%Y-%m-%d")
        opens, closes = data.open.loc[day], data.close.loc[day]
        for order in pending:
            fill = broker.execute(order, float(opens[order.symbol]), label)
            if fill:
                fills.append(fill)
        pending = []

        equity = broker.equity(closes)
        rows.append({"date": day, "equity": equity, "cash": broker.cash})
        peak = max(peak, equity)
        drawdown = 1 - equity / peak if peak > 0 else 0.0

        weights = current_weights(broker.positions, closes, equity)
        decision = decide(day, data, weights, drawdown)
        if decision is not None:
            decisions.append((day, decision))
            pending = plan_orders(
                decision.weights,
                broker.positions,
                closes,
                equity,
                cfg.risk.rebalance_band,
                cfg.broker.fractional,
            )

    equity_df = pd.DataFrame(rows, columns=["date", "equity", "cash"]).set_index("date")
    return SimResult(equity_df, fills, decisions, dict(broker.positions), broker.cash, pending)


def backtest_decider(strategy: Strategy, cfg: Config) -> DecideFn:
    last: list[pd.Timestamp | None] = [None]

    def decide(day, data, current, drawdown):
        decision = decide_step(strategy, cfg, day, data, current, drawdown, last[0])
        if decision is not None:
            last[0] = day
        return decision

    return decide
