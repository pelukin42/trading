"""Cash/position accounting shared by the simulated brokers (backtest and local paper account)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import Fill, OrderRequest, Position, Side


@dataclass
class Ledger:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    stops: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return self.cash + sum(p.qty * prices.get(s, p.market_price) for s, p in self.positions.items())

    def apply_fill(self, order_id: str, req: OrderRequest, price: float, t: datetime,
                   commission_bps: float, reason: str = "order") -> float:
        """Book a fill. Buys are capped by cash, sells by the shares held. Returns the filled quantity."""
        rate = commission_bps / 10_000
        pos = self.positions.get(req.symbol)
        if req.side == Side.BUY:
            qty = min(req.qty, max(self.cash / (price * (1 + rate)), 0.0))
        else:
            qty = min(req.qty, pos.qty if pos else 0.0)
        if qty <= 1e-9:
            return 0.0

        commission = qty * price * rate
        realized = 0.0
        if req.side == Side.BUY:
            self.cash -= qty * price + commission
            if pos:
                pos.avg_price = (pos.avg_price * pos.qty + price * qty) / (pos.qty + qty)
                pos.qty += qty
                pos.market_price = price
            else:
                self.positions[req.symbol] = Position(req.symbol, qty, price, price)
            if req.stop_loss_price:
                self.stops[req.symbol] = req.stop_loss_price
        else:
            self.cash += qty * price - commission
            realized = (price - pos.avg_price) * qty - commission
            pos.qty -= qty
            if pos.qty <= 1e-9:
                del self.positions[req.symbol]
                self.stops.pop(req.symbol, None)
        self.fills.append(Fill(order_id, req.symbol, req.side, qty, price, t, commission, realized, reason))
        return qty

    # --- persistence ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": {s: {"qty": p.qty, "avg_price": p.avg_price, "market_price": p.market_price}
                          for s, p in self.positions.items()},
            "stops": dict(self.stops),
            "fills": [{"order_id": f.order_id, "symbol": f.symbol, "side": f.side.value, "qty": f.qty,
                       "price": f.price, "t": f.t.isoformat(), "commission": f.commission,
                       "realized_pnl": f.realized_pnl, "reason": f.reason} for f in self.fills],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ledger":
        return cls(
            cash=d["cash"],
            positions={s: Position(s, p["qty"], p["avg_price"], p["market_price"]) for s, p in d["positions"].items()},
            stops=dict(d.get("stops", {})),
            fills=[Fill(f["order_id"], f["symbol"], Side(f["side"]), f["qty"], f["price"],
                        datetime.fromisoformat(f["t"]), f["commission"], f["realized_pnl"], f["reason"])
                   for f in d.get("fills", [])],
        )
