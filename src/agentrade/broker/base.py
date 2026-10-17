"""Broker interface. The agent, router and backtester only depend on this protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..models import Account, Bar, Order, OrderRequest, Position


class BrokerError(RuntimeError):
    pass


@dataclass
class Clock:
    now: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None


class Broker(Protocol):
    name: str

    def clock(self) -> Clock: ...

    def get_account(self) -> Account: ...

    def get_positions(self) -> dict[str, Position]: ...

    def get_open_orders(self) -> list[Order]:
        """Open entry/exit orders. Protective stop legs are excluded: they don't change target exposure."""
        ...

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        """Daily bars up to the broker's current time, oldest first."""
        ...

    def get_latest_price(self, symbol: str) -> float: ...

    def submit_order(self, order: OrderRequest) -> Order: ...

    def cancel_all_orders(self) -> None: ...

    def close_all_positions(self) -> None: ...
