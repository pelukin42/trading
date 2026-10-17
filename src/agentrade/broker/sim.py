"""In-memory simulated broker driven by historical bars.

Decisions are made with data up to the current bar's close; market orders fill at the
*next* bar's open (plus slippage), so backtests can't trade on prices they haven't seen.
"""

from __future__ import annotations

import itertools
from datetime import datetime

from ..ledger import Ledger
from ..models import Account, Bar, Order, OrderRequest, OrderStatus, OrderType, Position, Side
from .base import BrokerError, Clock


class SimBroker:
    name = "sim"

    def __init__(self, history: dict[str, list[Bar]], initial_cash: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 5.0, start_index: int = 0):
        if not history:
            raise BrokerError("SimBroker needs at least one symbol of history")
        self.history = {s.upper(): sorted(b, key=lambda x: x.t) for s, b in history.items()}
        self.timeline: list[datetime] = sorted({b.t for bars in self.history.values() for b in bars})
        self._by_time = {s: {b.t: b for b in bars} for s, bars in self.history.items()}
        self._index_of = {s: {b.t: i for i, b in enumerate(bars)} for s, bars in self.history.items()}
        self.cursor = start_index
        self.ledger = Ledger(cash=initial_cash)
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.pending: list[Order] = []
        self.orders: list[Order] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._ids = itertools.count(1)
        self._day_start_equity = initial_cash
        self._mark()

    # --- time -------------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self.ledger.cash

    @property
    def positions(self) -> dict[str, Position]:
        return self.ledger.positions

    @property
    def stops(self) -> dict[str, float]:
        return self.ledger.stops

    @property
    def fills(self):
        return self.ledger.fills

    @property
    def now(self) -> datetime:
        return self.timeline[self.cursor]

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.timeline) - 1

    def clock(self) -> Clock:
        return Clock(now=self.now, is_open=True)

    def advance(self) -> bool:
        """Move to the next bar, fill pending orders and trigger stops. Returns False at the end."""
        if self.finished:
            return False
        self._day_start_equity = self._equity()
        self.cursor += 1
        self._process_pending()
        self._process_stops()
        self._mark()
        return True

    # --- market data ------------------------------------------------------------------

    def _bar_now(self, symbol: str) -> Bar | None:
        return self._by_time.get(symbol, {}).get(self.now)

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        symbol = symbol.upper()
        bars = self.history.get(symbol)
        if bars is None:
            raise BrokerError(f"No data for {symbol}")
        # Last bar at or before now.
        idx = None
        for t in reversed(self.timeline[: self.cursor + 1]):
            if t in self._index_of[symbol]:
                idx = self._index_of[symbol][t]
                break
        if idx is None:
            return []
        return bars[max(0, idx + 1 - limit): idx + 1]

    def get_latest_price(self, symbol: str) -> float:
        bars = self.get_bars(symbol, limit=1)
        if not bars:
            raise BrokerError(f"No price yet for {symbol}")
        return bars[-1].close

    # --- account ----------------------------------------------------------------------

    def _equity(self) -> float:
        return self.ledger.equity()

    def _mark(self) -> None:
        for s, p in self.positions.items():
            bar = self._bar_now(s)
            if bar:
                p.market_price = bar.close
        self.equity_curve.append((self.now, self._equity()))

    def get_account(self) -> Account:
        equity = self._equity()
        return Account(equity=equity, cash=self.cash, buying_power=max(self.cash, 0.0),
                       day_start_equity=self._day_start_equity)

    def get_positions(self) -> dict[str, Position]:
        return {s: Position(p.symbol, p.qty, p.avg_price, p.market_price) for s, p in self.positions.items()}

    def get_open_orders(self) -> list[Order]:
        return list(self.pending)

    # --- orders -----------------------------------------------------------------------

    def submit_order(self, order: OrderRequest) -> Order:
        o = Order(id=f"sim-{next(self._ids)}", request=order, status=OrderStatus.PENDING, submitted_at=self.now)
        self.pending.append(o)
        self.orders.append(o)
        return o

    def cancel_all_orders(self) -> None:
        for o in self.pending:
            o.status = OrderStatus.CANCELED
        self.pending = []

    def close_all_positions(self) -> None:
        self.cancel_all_orders()
        for s, p in list(self.positions.items()):
            self.submit_order(OrderRequest(symbol=s, side=Side.SELL, qty=p.qty, rationale="close all"))

    def _process_pending(self) -> None:
        # Sells first so their proceeds can fund buys queued in the same step.
        queue = sorted(self.pending, key=lambda o: o.request.side != Side.SELL)
        still_pending: list[Order] = []
        for o in queue:
            bar = self._bar_now(o.symbol)
            if bar is None:
                still_pending.append(o)
                continue
            price = self._fill_price(o, bar)
            if price is None:
                if o.request.order_type == OrderType.LIMIT:
                    o.status = OrderStatus.CANCELED  # day orders expire if not filled on the next bar
                    continue
                still_pending.append(o)
                continue
            self._fill(o, price)
        self.pending = still_pending

    def _fill_price(self, o: Order, bar: Bar) -> float | None:
        slip = self.slippage_bps / 10_000
        if o.request.order_type == OrderType.MARKET:
            return bar.open * (1 + slip) if o.request.side == Side.BUY else bar.open * (1 - slip)
        limit = o.request.limit_price or 0.0
        if o.request.side == Side.BUY and bar.low <= limit:
            return min(bar.open, limit)
        if o.request.side == Side.SELL and bar.high >= limit:
            return max(bar.open, limit)
        return None

    def _fill(self, o: Order, price: float, reason: str = "order") -> None:
        qty = self.ledger.apply_fill(o.id, o.request, price, self.now, self.commission_bps, reason)
        if qty <= 0:
            o.status = OrderStatus.REJECTED
            return
        o.filled_qty = qty
        o.filled_avg_price = price
        o.status = OrderStatus.FILLED if qty >= o.request.qty - 1e-9 else OrderStatus.PARTIALLY_FILLED

    def _process_stops(self) -> None:
        for symbol, stop in list(self.stops.items()):
            pos = self.positions.get(symbol)
            bar = self._bar_now(symbol)
            if not pos or not bar or bar.low > stop:
                continue
            price = min(bar.open, stop) * (1 - self.slippage_bps / 10_000)  # gaps fill at the open
            o = Order(id=f"sim-{next(self._ids)}",
                      request=OrderRequest(symbol, Side.SELL, pos.qty, rationale=f"stop loss @ {stop:.2f}"),
                      status=OrderStatus.PENDING, submitted_at=self.now)
            self.orders.append(o)
            self._fill(o, price, reason="stop")
            # Pending orders for a stopped-out symbol would reopen the position blindly; drop them.
            for p in self.pending:
                if p.symbol == symbol:
                    p.status = OrderStatus.CANCELED
            self.pending = [p for p in self.pending if p.symbol != symbol]
