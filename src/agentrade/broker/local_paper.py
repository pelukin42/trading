"""Paper-trading account kept on disk, priced with real market data. No broker account needed.

Whenever the account is read or traded, it first replays every session since it was last looked
at, oldest first: orders queued for a session fill at that session's open, then stops are checked
against that session's daily range. Only then is the new day's P&L baseline taken and the session
in progress handled. Fill rules, chosen to be realistic without intraday data:
- Market orders while NYSE is open fill at the latest quote plus slippage, provided the quote was
  printed in the current session (before the first print they wait for the session's open price).
  Orders placed while the market is closed fill at the next session's open.
- Limit orders are day orders: placed during a session they fill only if a later quote crosses the
  limit that same session; queued for the next session they fill if that day's range reaches the limit.
- Stops trigger on a live quote at or below the stop, or on a final daily bar whose low reached it
  (filling at the open when the price gapped through). The bar of the day a position was entered
  or added to intraday is skipped, because its low may have come before the entry; when adding
  intraday, the day's range so far is first checked against the existing stop.
- A stop-out cancels the symbol's other open orders, as do most brokers' OCO-style protections.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .. import market_calendar as cal
from ..ledger import Ledger
from ..models import Account, Bar, Fill, Order, OrderRequest, OrderStatus, OrderType, Position, Side
from ..yahoo import Quote
from .base import BrokerError, Clock

try:
    import fcntl
except ImportError:  # Windows: single-process use only
    fcntl = None

# After the close, wait this long before treating the day's bar as final.
SETTLE = timedelta(minutes=20)
MAX_EVENTS = 200


class MarketData(Protocol):
    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]: ...

    def get_quote(self, symbol: str) -> Quote: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _seq(p: dict[str, Any]) -> int:
    return int(p["id"].split("-")[1])


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
        self.events: list[dict[str, Any]] = []  # orders that expired, were canceled or rejected
        self.next_id = 1

    def _load(self) -> None:
        d = json.loads(self.state_path.read_text())
        self.initial_cash = d["initial_cash"]
        self.created_at = d["created_at"]
        self.ledger = Ledger.from_dict(d["ledger"])
        self.pending = d["pending"]
        self.stop_checked = d["stop_checked"]
        self.day = d["day"]
        self.events = d.get("events", [])
        self.next_id = d["next_id"]

    def _save(self) -> None:
        d = {"version": 1, "created_at": self.created_at, "initial_cash": self.initial_cash,
             "ledger": self.ledger.to_dict(), "pending": self.pending, "stop_checked": self.stop_checked,
             "day": self.day, "events": self.events[-MAX_EVENTS:], "next_id": self.next_id}
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
        """Reload state (another process may have changed it), catch up on missed sessions, then save."""
        with self._file_lock():
            self._load()
            now = self._now()
            current = cal.last_session_started(now)
            start = self._replay_from()
            if start is not None:
                d = start
                while d < current:
                    if cal.is_trading_day(d):
                        self._process_session(d, now)
                    d += timedelta(days=1)
            self._roll_day(current)
            self._process_session(current, now)
            yield now
            self._save()

    def reset(self, cash: float) -> None:
        with self._file_lock():
            self._init_state(cash)
            self._save()

    # --- helpers ----------------------------------------------------------------------

    def _new_id(self) -> str:
        oid = f"lp-{self.next_id}"
        self.next_id += 1
        return oid

    def _slipped(self, price: float, side: Side) -> float:
        slip = self.slippage_bps / 10_000
        return price * (1 + slip) if side == Side.BUY else price * (1 - slip)

    def _session_bar(self, symbol: str, d: date, now: datetime) -> Bar | None:
        # Ask for enough bars to reach back to `d` (calendar days >= trading days).
        limit = max(10, (cal.session_date(now) - d).days + 10)
        for b in self.data.get_bars(symbol, limit=limit):
            if cal.session_date(b.t) == d:
                return b
        return None

    @staticmethod
    def _bar_is_final(d: date, now: datetime) -> bool:
        return now >= cal.session_bounds(d)[1] + SETTLE

    def _session_quote(self, symbol: str, now: datetime) -> float | None:
        """Latest price if NYSE is open and it was printed in the session in progress, else None.

        Before the first print of the day the feed still reports the previous close.
        """
        if not cal.is_open(now):
            return None
        q = self.data.get_quote(symbol)
        if q.t < cal.session_bounds(cal.last_session_started(now))[0]:
            return None
        return q.price

    def _event(self, p: dict[str, Any], status: str, reason: str, now: datetime) -> None:
        self.events.append({"t": now.isoformat(), "id": p["id"], "symbol": p["symbol"], "side": p["side"],
                            "qty": p["qty"], "status": status, "reason": reason})
        self.events = self.events[-MAX_EVENTS:]

    def _replay_from(self) -> date | None:
        dates = [date.fromisoformat(p["session"]) for p in self.pending]
        dates += [date.fromisoformat(self.stop_checked[s]) + timedelta(days=1)
                  for s in self.ledger.stops if s in self.stop_checked]
        return min(dates) if dates else None

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

    # --- session processing -----------------------------------------------------------

    def _roll_day(self, session: date) -> None:
        """On the first look at a new session, record equity at the previous close for day P&L."""
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

    def _process_session(self, d: date, now: datetime) -> None:
        self._process_pending(d, now)
        self._process_stops(d, now)

    def _execution_order(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Submission order, except that a sell moves ahead of buys (so its proceeds can fund them)
        unless an earlier buy of the same symbol is queued before it."""
        ordered = sorted(orders, key=_seq)
        first_buy: dict[str, int] = {}
        for p in ordered:
            if p["side"] == "buy":
                first_buy.setdefault(p["symbol"], _seq(p))
        early = [p for p in ordered if p["side"] == "sell" and _seq(p) < first_buy.get(p["symbol"], math.inf)]
        return early + [p for p in ordered if p not in early]

    def _process_pending(self, d: date, now: datetime) -> None:
        todo = self._execution_order([p for p in self.pending if p["session"] == d.isoformat()])
        for p in todo:
            if p not in self.pending:
                continue  # canceled by a stop-out earlier in this pass
            try:
                outcome = self._try_fill(p, d, now)
            except BrokerError:
                continue  # data unavailable: try again on the next look
            if outcome is None:
                continue  # still working
            self.pending.remove(p)
            if outcome != "filled":
                self._event(p, *outcome, now)

    def _try_fill(self, p: dict[str, Any], d: date, now: datetime) -> str | tuple[str, str] | None:
        """Returns "filled", (status, reason) when the order is done without a fill, or None to keep it."""
        req = self._request(p)
        start, end = cal.session_bounds(d)
        final = self._bar_is_final(d, now)

        if p["live_only"]:
            # A limit order placed mid-session: only quotes seen after it count.
            if now >= end:
                return ("expired", "day limit order not reached")
            if req.side == Side.BUY:
                if self._stopped_before_add(req.symbol, d, now):
                    return None  # the stop-out already canceled this order
            q = self._session_quote(req.symbol, now)
            if q is None or not ((req.side == Side.BUY and q <= req.limit_price) or
                                 (req.side == Side.SELL and q >= req.limit_price)):
                return None
            return self._fill_or_reject(p["id"], req, q, now, d, intraday=True)

        bar = self._session_bar(req.symbol, d, now)
        if req.order_type == OrderType.MARKET:
            if bar is None:
                return ("expired", f"no trading data for {req.symbol} on {d}") if final else None
            return self._fill_or_reject(p["id"], req, self._slipped(bar.open, req.side), start, d, intraday=False)

        price = None
        if bar is not None and req.side == Side.BUY and bar.low <= req.limit_price:
            price = min(bar.open, req.limit_price)
        elif bar is not None and req.side == Side.SELL and bar.high >= req.limit_price:
            price = max(bar.open, req.limit_price)
        if price is not None:
            return self._fill_or_reject(p["id"], req, price, min(now, end), d, intraday=False)
        return ("expired", "day limit order not reached") if final else None

    def _fill_or_reject(self, oid: str, req: OrderRequest, price: float, t: datetime, d: date,
                        intraday: bool) -> str | tuple[str, str]:
        if self._book(oid, req, price, t, d, intraday) > 0:
            return "filled"
        return ("rejected", "no shares to sell" if req.side == Side.SELL else "not enough cash")

    def _stop_out(self, symbol: str, price: float, t: datetime, d: date, intraday: bool, now: datetime) -> None:
        pos = self.ledger.positions[symbol]
        stop = self.ledger.stops[symbol]
        req = OrderRequest(symbol, Side.SELL, pos.qty, rationale=f"stop loss @ {stop:.2f}")
        self._book(self._new_id(), req, self._slipped(price, Side.SELL), t, d, intraday, reason="stop")
        for p in [p for p in self.pending if p["symbol"] == symbol]:
            self.pending.remove(p)
            self._event(p, "canceled", f"{symbol} was stopped out at {stop:.2f}", now)

    def _stopped_before_add(self, symbol: str, d: date, now: datetime) -> bool:
        """Before adding to a position mid-session, check the day's range so far against its stop."""
        stop = self.ledger.stops.get(symbol)
        if stop is None or symbol not in self.ledger.positions:
            return False
        checked = self.stop_checked.get(symbol)
        if checked and date.fromisoformat(checked) >= d:
            return False  # an earlier intraday entry today: the range so far predates it
        bar = self._session_bar(symbol, d, now)
        if bar is None or bar.low > stop:
            return False
        self._stop_out(symbol, min(bar.open, stop), now, d, intraday=True, now=now)
        return True

    def _process_stops(self, d: date, now: datetime) -> None:
        for symbol, stop in list(self.ledger.stops.items()):
            if symbol not in self.ledger.positions:
                continue
            checked = self.stop_checked.get(symbol)
            try:
                if not (checked and date.fromisoformat(checked) >= d) and self._bar_is_final(d, now):
                    bar = self._session_bar(symbol, d, now)
                    self.stop_checked[symbol] = d.isoformat()
                    if bar is not None and bar.low <= stop:
                        self._stop_out(symbol, min(bar.open, stop), cal.session_bounds(d)[1], d,
                                       intraday=False, now=now)
                        continue
                if d == cal.last_session_started(now):
                    q = self._session_quote(symbol, now)
                    if q is not None and q <= stop:
                        self._stop_out(symbol, q, now, d, intraday=True, now=now)
            except BrokerError:
                continue

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

    def get_fills(self) -> list[Fill]:
        with self._transaction():
            return list(self.ledger.fills)

    def recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """Orders that expired, were canceled or were rejected, newest last."""
        with self._transaction():
            return self.events[-limit:]

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
            if order.side == Side.BUY and self._stopped_before_add(order.symbol, session, now):
                self._event(pending, "rejected", f"{order.symbol} was stopped out earlier today", now)
                return Order(id=oid, request=order, status=OrderStatus.REJECTED, submitted_at=now)

            q = self._session_quote(order.symbol, now)
            if q is None:
                # No print yet this session: a market order takes the opening price, a limit order waits.
                pending.update(session=session.isoformat(), live_only=order.order_type == OrderType.LIMIT)
                self.pending.append(pending)
                return self._to_order(pending)

            if order.order_type == OrderType.MARKET:
                price = self._slipped(q, order.side)
            elif (order.side == Side.BUY and q <= order.limit_price) or \
                    (order.side == Side.SELL and q >= order.limit_price):
                price = q
            else:
                pending.update(session=session.isoformat(), live_only=True)
                self.pending.append(pending)
                return self._to_order(pending)

            qty = self._book(oid, order, price, now, session, intraday=True)
            if qty <= 0:
                self._event(pending, "rejected", "not enough cash" if order.side == Side.BUY else "no shares to sell",
                            now)
            status = (OrderStatus.REJECTED if qty <= 0 else
                      OrderStatus.FILLED if qty >= order.qty - 1e-9 else OrderStatus.PARTIALLY_FILLED)
            return Order(id=oid, request=order, status=status, submitted_at=now, filled_qty=qty,
                         filled_avg_price=price if qty > 0 else None)

    def cancel_all_orders(self) -> None:
        with self._transaction() as now:
            for p in self.pending:
                self._event(p, "canceled", "canceled by user", now)
            self.pending = []

    def close_all_positions(self) -> None:
        self.cancel_all_orders()
        with self._transaction():
            positions = list(self.ledger.positions.values())
        for p in positions:
            self.submit_order(OrderRequest(p.symbol, Side.SELL, p.qty, rationale="close all"))
