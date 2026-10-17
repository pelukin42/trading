"""Core domain types shared by brokers, risk, agent and backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Bar:
    t: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    # Equity at the previous session close; daily loss limits are measured against it.
    day_start_equity: float
    currency: str = "USD"

    @property
    def day_pnl(self) -> float:
        return self.equity - self.day_start_equity

    @property
    def day_pnl_pct(self) -> float:
        return self.day_pnl / self.day_start_equity if self.day_start_equity else 0.0


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.market_price - self.avg_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        return self.market_price / self.avg_price - 1 if self.avg_price else 0.0


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    # Protective stop attached to an entry; the broker places it once the entry fills.
    stop_loss_price: float | None = None
    rationale: str = ""


@dataclass
class Order:
    id: str
    request: OrderRequest
    status: OrderStatus
    submitted_at: datetime
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    reason: str | None = None  # why the broker rejected or canceled it, when it says

    @property
    def symbol(self) -> str:
        return self.request.symbol

    @property
    def remaining_qty(self) -> float:
        return self.request.qty - self.filled_qty


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    t: datetime
    commission: float = 0.0
    realized_pnl: float = 0.0
    reason: str = "order"  # "order" | "stop"


@dataclass(frozen=True)
class CorporateAction:
    """A split (ratio = new shares per old share) or a cash dividend (amount per share), effective
    at the open of `session`. Amounts are expressed per share as of today, like split-adjusted prices."""

    session: date
    kind: str  # "split" | "dividend"
    ratio: float = 1.0
    amount: float = 0.0


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "RiskDecision":
        return cls(True, [])
