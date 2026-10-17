"""Tools exposed to the agent and their execution against the router/broker."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from .. import indicators as ind
from ..broker.base import Broker, BrokerError
from ..config import RiskLimits
from ..models import OrderRequest, OrderType, Side
from ..router import OrderRouter

MAX_HISTORY_DAYS = 120
log = logging.getLogger("agentrade")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_portfolio",
        "description": "Account equity, cash, today's P&L, open positions with unrealized P&L, "
                       "pending orders, and how much room is left under each risk limit.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "get_market_data",
        "description": "Indicator snapshot per symbol from daily bars: last close, 1/5/20/60-day returns, "
                       "SMA 20/50/200, RSI(14), ATR(14), 20-day annualized volatility, distance from the "
                       "52-week high/low, volume vs. its 20-day average, and the last 10 closes.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}, "description": "Ticker symbols."},
            },
            "required": ["symbols"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_price_history",
        "description": f"Daily OHLCV bars for one symbol, oldest first, up to {MAX_HISTORY_DAYS} days. "
                       "Use only when the snapshot from get_market_data isn't enough.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days": {"type": "integer", "description": f"Number of bars, 5 to {MAX_HISTORY_DAYS}."},
            },
            "required": ["symbol", "days"],
            "additionalProperties": False,
        },
    },
    {
        "name": "place_order",
        "description": "Submit an order. It is checked by the risk engine first; the result says whether it "
                       "was approved and, if not, exactly why. Quantities are rounded down to whole shares.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "quantity": {"type": "number", "description": "Amount, in the unit given by quantity_type."},
                "quantity_type": {"type": "string", "enum": ["shares", "usd"]},
                "order_type": {"type": "string", "enum": ["market", "limit"]},
                "limit_price": {"type": "number", "description": "Required for limit orders."},
                "stop_loss_price": {"type": "number",
                                    "description": "Protective stop below entry; required for buys."},
                "rationale": {"type": "string", "description": "Why this trade, in one or two sentences."},
            },
            "required": ["symbol", "side", "quantity", "quantity_type", "order_type", "rationale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "close_position",
        "description": "Sell the entire position in a symbol at market.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "rationale": {"type": "string"}},
            "required": ["symbol", "rationale"],
            "additionalProperties": False,
        },
    },
]

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def market_snapshot(broker: Broker, symbol: str) -> dict[str, Any]:
    bars = broker.get_bars(symbol, limit=260)
    if len(bars) < 2:
        return {"symbol": symbol, "error": "not enough history"}
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    last = bars[-1]
    year = closes[-252:]
    atr14 = ind.atr(bars, 14)
    avg_vol = ind.sma(volumes[:-1], 20)
    return {
        "symbol": symbol,
        "as_of": last.t.date().isoformat(),
        "last_close": _r(last.close, 2),
        "return_1d": _r(ind.pct_return(closes, 1)),
        "return_5d": _r(ind.pct_return(closes, 5)),
        "return_20d": _r(ind.pct_return(closes, 20)),
        "return_60d": _r(ind.pct_return(closes, 60)),
        "sma_20": _r(ind.sma(closes, 20), 2),
        "sma_50": _r(ind.sma(closes, 50), 2),
        "sma_200": _r(ind.sma(closes, 200), 2),
        "rsi_14": _r(ind.rsi(closes, 14), 1),
        "atr_14": _r(atr14, 2),
        "atr_pct": _r(atr14 / last.close if atr14 else None),
        "volatility_20d_annual": _r(ind.annualized_volatility(closes, 20)),
        "pct_from_52w_high": _r(last.close / max(year) - 1),
        "pct_from_52w_low": _r(last.close / min(year) - 1),
        "volume_vs_20d_avg": _r(last.volume / avg_vol if avg_vol else None, 2),
        "last_10_closes": [round(c, 2) for c in closes[-10:]],
    }


class ToolExecutor:
    def __init__(self, broker: Broker, router: OrderRouter, limits: RiskLimits, watchlist: list[str]):
        self.broker = broker
        self.router = router
        self.limits = limits
        self.watchlist = [s.strip().upper() for s in watchlist]

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Returns (json_result, is_error)."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool {name}"}), True
        try:
            return json.dumps(handler(**args), sort_keys=True, ensure_ascii=False), False
        except (BrokerError, ValueError, TypeError, KeyError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
        except Exception as e:  # a tool bug must not take down the trading loop
            log.exception("Tool %s failed", name)
            return json.dumps({"error": f"Internal error in {name}: {type(e).__name__}"}), True

    def _tool_get_portfolio(self) -> dict[str, Any]:
        account, positions, open_orders = self.broker.snapshot()
        L = self.limits
        gross = sum(abs(p.market_value) for p in positions.values())
        eq = account.equity or 1.0
        return {
            "equity": round(account.equity, 2),
            "cash": round(account.cash, 2),
            "day_pnl": round(account.day_pnl, 2),
            "day_pnl_pct": round(account.day_pnl_pct, 4),
            "gross_exposure_pct": round(gross / eq, 4),
            "positions": [
                {"symbol": p.symbol, "qty": p.qty, "avg_price": round(p.avg_price, 2),
                 "market_price": round(p.market_price, 2), "market_value": round(p.market_value, 2),
                 "weight": round(p.market_value / eq, 4), "unrealized_pnl": round(p.unrealized_pnl, 2),
                 "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4)}
                for p in sorted(positions.values(), key=lambda p: -abs(p.market_value))
            ],
            "open_orders": [
                {"id": o.id, "symbol": o.symbol, "side": o.request.side.value, "qty": o.remaining_qty,
                 "type": o.request.order_type.value, "limit_price": o.request.limit_price}
                for o in open_orders
            ],
            "risk_room": {
                "max_position_usd": round(L.max_position_pct * account.equity, 2),
                "gross_exposure_left_usd": round(max(L.max_gross_exposure_pct * account.equity - gross, 0), 2),
                "open_position_slots_left": max(L.max_open_positions - len(positions), 0),
                "max_order_usd": L.max_order_notional,
                "daily_loss_limit_pct": L.max_daily_loss_pct,
                "new_risk_allowed_today": account.day_pnl_pct > -L.max_daily_loss_pct,
            },
            "kill_switch_active": self.router.risk.kill_switch_active(),
        }

    def _tool_get_market_data(self, symbols: list[str]) -> dict[str, Any]:
        out = []
        for s in symbols[:20]:
            try:
                out.append(market_snapshot(self.broker, s.strip().upper()))
            except BrokerError as e:
                out.append({"symbol": s.strip().upper(), "error": str(e)})
        return {"snapshots": out}

    def _tool_get_price_history(self, symbol: str, days: int) -> dict[str, Any]:
        days = min(max(int(days), 5), MAX_HISTORY_DAYS)
        bars = self.broker.get_bars(symbol.strip().upper(), limit=days)
        return {
            "symbol": symbol.strip().upper(),
            "columns": ["date", "open", "high", "low", "close", "volume"],
            "bars": [[b.t.date().isoformat(), round(b.open, 2), round(b.high, 2), round(b.low, 2),
                      round(b.close, 2), int(b.volume)] for b in bars],
        }

    def _tool_place_order(self, symbol: str, side: str, quantity: float, quantity_type: str, order_type: str,
                          rationale: str, limit_price: float | None = None,
                          stop_loss_price: float | None = None) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        otype = OrderType(order_type)
        if otype == OrderType.LIMIT and not limit_price:
            raise ValueError("limit orders need limit_price")
        if quantity_type == "usd":
            price = limit_price if otype == OrderType.LIMIT else self.broker.get_latest_price(symbol)
            shares = math.floor(quantity / price)
        else:
            shares = math.floor(quantity)
        if shares < 1:
            raise ValueError(f"order rounds to {shares} shares; increase the quantity")
        req = OrderRequest(symbol=symbol, side=Side(side), qty=shares, order_type=otype,
                           limit_price=limit_price if otype == OrderType.LIMIT else None,
                           stop_loss_price=stop_loss_price if side == "buy" else None, rationale=rationale)
        return self.router.route(req).to_dict()

    def _tool_close_position(self, symbol: str, rationale: str) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        pos = self.broker.get_positions().get(symbol)
        if not pos or pos.qty <= 0:
            raise ValueError(f"no open position in {symbol}")
        req = OrderRequest(symbol=symbol, side=Side.SELL, qty=pos.qty, rationale=rationale)
        return self.router.route(req).to_dict()
