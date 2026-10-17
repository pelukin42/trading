"""Paper-trading account kept on disk, priced with real market data. No broker account needed.

Fill rules, chosen to be realistic without intraday data:
- Market orders while NYSE is open fill immediately at the latest quote plus slippage.
- Orders placed while it is closed wait for the next session and fill at that session's open.
- Limit orders are day orders: placed during a session they fill only if a later quote crosses the
  limit that same session; queued for the next session they fill if that day's range reaches the limit.
- Stops trigger on a live quote at or below the stop, or on a completed daily bar whose low reached
  it (filling at the open when the price gapped through). The bar of the day a position was entered
  intraday is skipped, because its low may have come before the entry.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .. import market_calendar as cal
from ..ledger import Ledger
from ..models import Account, Bar, Order, OrderRequest, OrderStatus, OrderType, Position, Side
from ..yahoo import Quote
from .base import BrokerError, Clock

try:
    import fcntl
except ImportError:  # Windows: single-process use only
    fcntl = None


class MarketData(Protocol):
    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]: ...

    def get_quote(self, symbol: str) -> Quote: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LocalPaperBroker:
    name = "local-paper"

    def __init__(self, data: MarketData, state_path: str | Path, initial_cash: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 5.0,
                 now: Callable[[], datetime] = _utcnow):
        self.data = data
        self.state_path = Path(state_path)
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self._now = now
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            if self.state_path.exists():
                self._load()
            else:
                self._init_state(initial_cash)
                self._save()

    # --- persistence ------------------------------------------------------------------

    def _init_state(self, cash: float) -> None:
        self.initial_cash = cash
        self.created_at = self._now().isoformat()
        self.ledger = Ledger(cash=cash)
        self.pending: list[dict[str, Any]] = []
        # Last session whose daily bar has been checked against each symbol's stop.
        self.stop_checked: dict[str, str] = {}
        self.day: dict[str, Any] = {"session": None, "start_equity": cash}
        self.next_id = 1

    def _load(self) -> None:
        d = json.loads(self.state_path.read_text())
        self.initial_cash = d["initial_cash"]
        self.created_at = d["created_at"]
        self.ledger = Ledger.from_dict(d["ledger"])
        self.pending = d["pending"]
        self.stop_checked = d["stop_checked"]
        self.day = d["day"]
        self.next_id = d["next_id"]

    def _save(self) -> None:
        d = {"version": 1, "created_at": self.created_at, "initial_cash": self.initial_cash,
             "ledger": self.ledger.to_dict(), "pending": self.pending, "stop_checked": self.stop_checked,
             "day": self.day, "next_id": self.next_id}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, self.state_path)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        if fcntl is None:
            yield
            return
        with open(self.state_path.with_suffix(".lock"), "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @contextmanager
    def _transaction(self) -> Iterator[datetime]:
        """Reload state (another process may have changed it), catch up on fills, then save."""
        with self._file_lock():
            self._load()
            now = self._now()
            self._roll_day(now)
            self._process_pending(now)
            self._process_stops(now)
            yield now
            self._save()

    def reset(self, cash: float) -> None:
        with self._file_lock():
            self._init_state(cash)
            self._save()

    # --- helpers ----------------------------------------------------------------------

    def _slipped(self, price: float, side: Side) -> float:
        slip = self.slippage_bps / 10_000
        return price * (1 + slip) if side == Side.BUY else price * (1 - slip)

    def _session_bar(self, symbol: str, d: date) -> Bar | None:
        for b in self.data.get_bars(symbol, limit=10):
            if cal.session_date(b.t) == d:
                return b
        return None

    def _book(self, order_id: str, req: OrderRequest, price: float, t: datetime, entry_session: date,
              intraday: bool, reason: str = "order") -> float:
        qty = self.ledger.apply_fill(order_id, req, price, t, self.commission_bps, reason)
        if qty > 0 and req.side == Side.BUY and req.stop_loss_price:
            checked = entry_session if intraday else entry_session - timedelta(days=1)
            self.stop_checked[req.symbol] = checked.isoformat()
        return qty

    @staticmethod
    def _request(p: dict[str, Any]) -> OrderRequest:
        return OrderRequest(symbol=p["symbol"], side=Side(p["side"]), qty=p["qty"],
                            order_type=OrderType(p["order_type"]), limit_price=p["limit_price"],
                            stop_loss_price=p["stop_loss_price"], rationale=p["rationale"])

    def _to_order(self, p: dict[str, Any]) -> Order:
        return Order(id=p["id"], request=self._request(p), status=OrderStatus.PENDING,
                     submitted_at=datetime.fromisoformat(p["submitted_at"]))

    # --- catch-up processing ------------------------------------------------------------

    def _roll_day(self, now: datetime) -> None:
        """On the first look at a new session, record equity at the previous close for day P&L."""
        session = cal.last_session_started(now)
        if self.day["session"] == session.isoformat():
            return
        prices = {}
        for s in self.ledger.positions:
            try:
                prior = [b for b in self.data.get_bars(s, limit=10) if cal.session_date(b.t) < session]
                if prior:
                    prices[s] = prior[-1].close
            except BrokerError:
                pass
        self.day = {"session": session.isoformat(), "start_equity": self.ledger.equity(prices)}

    def _process_pending(self, now: datetime) -> None:
        still: list[dict[str, Any]] = []
        # Sells first, so their proceeds can fund buys queued for the same open.
        for p in sorted(self.pending, key=lambda p: p["side"] != "sell"):
            target = date.fromisoformat(p["session"])
            start, end = cal.session_bounds(target)
            if now < start:
                still.append(p)
                continue
            req = self._request(p)
            try:
                bar = self._session_bar(req.symbol, target)
            except BrokerError:
                still.append(p)  # data unavailable: try again on the next look
                continue

            if req.order_type == OrderType.MARKET:
                if bar is None:
                    still.append(p)  # the session's bar hasn't shown up in the data yet
                    continue
                self._book(p["id"], req, self._slipped(bar.open, req.side), start, target, intraday=False)
                continue

            price = None
            if p["live_only"]:
                if now < end:
                    try:
                        q = self.data.get_quote(req.symbol).price
                    except BrokerError:
                        q = None
                    if q is not None and req.side == Side.BUY and q <= req.limit_price:
                        price = q
                    elif q is not None and req.side == Side.SELL and q >= req.limit_price:
                        price = q
            elif bar is not None:
                if req.side == Side.BUY and bar.low <= req.limit_price:
                    price = min(bar.open, req.limit_price)
                elif req.side == Side.SELL and bar.high >= req.limit_price:
                    price = max(bar.open, req.limit_price)
            if price is not None:
                self._book(p["id"], req, price, min(now, end), target, intraday=bool(p["live_only"]))
            elif now < end:
                still.append(p)
            # else: a day order whose session ended unfilled expires
        self.pending = still

    def _process_stops(self, now: datetime) -> None:
        current = cal.last_session_started(now)
        open_now = cal.is_open(now)
        for symbol, stop in list(self.ledger.stops.items()):
            pos = self.ledger.positions.get(symbol)
            if not pos:
                continue
            req = OrderRequest(symbol, Side.SELL, pos.qty, rationale=f"stop loss @ {stop:.2f}")
            try:
                bars = self.data.get_bars(symbol, limit=10)
            except BrokerError:
                continue
            checked = self.stop_checked.get(symbol)
            triggered = False
            for b in bars:
                d = cal.session_date(b.t)
                if checked and d <= date.fromisoformat(checked):
                    continue
                complete = d < current or (d == current and now >= cal.session_bounds(d)[1])
                if not complete:
                    continue
                self.stop_checked[symbol] = d.isoformat()
                if b.low <= stop:
                    price = self._slipped(min(b.open, stop), Side.SELL)
                    self._book(self._new_id(), req, price, cal.session_bounds(d)[1], d, intraday=False,
                               reason="stop")
                    triggered = True
                    break
            if triggered or not open_now:
                continue
            try:
                q = self.data.get_quote(symbol).price
            except BrokerError:
                continue
            if q <= stop:
                self._book(self._new_id(), req, self._slipped(q, Side.SELL), now, current, intraday=True,
                           reason="stop")

    def _new_id(self) -> str:
        oid = f"lp-{self.next_id}"
        self.next_id += 1
        return oid

    # --- broker protocol --------------------------------------------------------------

    def clock(self) -> Clock:
        return cal.clock(self._now())

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        return self.data.get_bars(symbol, limit)

    def get_latest_price(self, symbol: str) -> float:
        return self.data.get_quote(symbol).price

    def _mark(self) -> None:
        for s, p in self.ledger.positions.items():
            try:
                p.market_price = self.data.get_quote(s).price
            except BrokerError:
                pass  # keep the last known price

    def get_account(self) -> Account:
        with self._transaction():
            self._mark()
            return Account(equity=self.ledger.equity(), cash=self.ledger.cash,
                           buying_power=max(self.ledger.cash, 0.0), day_start_equity=self.day["start_equity"])

    def get_positions(self) -> dict[str, Position]:
        with self._transaction():
            self._mark()
            return {s: Position(p.symbol, p.qty, p.avg_price, p.market_price)
                    for s, p in self.ledger.positions.items()}

    def get_open_orders(self) -> list[Order]:
        with self._transaction():
            return [self._to_order(p) for p in self.pending]

    def get_fills(self):
        with self._transaction():
            return list(self.ledger.fills)

    def submit_order(self, order: OrderRequest) -> Order:
        if order.qty <= 0:
            raise BrokerError("quantity must be positive")
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            raise BrokerError("limit orders need a limit_price")
        with self._transaction() as now:
            oid = self._new_id()
            pending = {"id": oid, "symbol": order.symbol, "side": order.side.value, "qty": order.qty,
                       "order_type": order.order_type.value, "limit_price": order.limit_price,
                       "stop_loss_price": order.stop_loss_price, "rationale": order.rationale,
                       "submitted_at": now.isoformat()}
            if not cal.is_open(now):
                pending.update(session=cal.next_session_after(now).isoformat(), live_only=False)
                self.pending.append(pending)
                return self._to_order(pending)

            session = cal.last_session_started(now)
            q = self.data.get_quote(order.symbol).price
            if order.order_type == OrderType.MARKET:
                price = self._slipped(q, order.side)
            elif order.side == Side.BUY and q <= order.limit_price:
                price = q
            elif order.side == Side.SELL and q >= order.limit_price:
                price = q
            else:
                pending.update(session=session.isoformat(), live_only=True)
                self.pending.append(pending)
                return self._to_order(pending)

            qty = self._book(oid, order, price, now, session, intraday=True)
            status = (OrderStatus.REJECTED if qty <= 0 else
                      OrderStatus.FILLED if qty >= order.qty - 1e-9 else OrderStatus.PARTIALLY_FILLED)
            return Order(id=oid, request=order, status=status, submitted_at=now, filled_qty=qty,
                         filled_avg_price=price if qty > 0 else None)

    def cancel_all_orders(self) -> None:
        with self._transaction():
            self.pending = []

    def close_all_positions(self) -> None:
        self.cancel_all_orders()
        with self._transaction():
            positions = list(self.ledger.positions.values())
        for p in positions:
            self.submit_order(OrderRequest(p.symbol, Side.SELL, p.qty, rationale="close all"))
