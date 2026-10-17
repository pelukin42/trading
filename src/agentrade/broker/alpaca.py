"""Alpaca adapter (https://docs.alpaca.markets). Paper and live share the same API on different hosts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx2

from ..models import Account, Bar, Order, OrderRequest, OrderStatus, OrderType, Position, Side
from .base import BrokerError, Clock

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

_PROTECTIVE_TYPES = {"stop", "stop_limit", "trailing_stop"}
_STATUS = {
    "new": OrderStatus.PENDING, "accepted": OrderStatus.PENDING, "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING, "held": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED, "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED, "expired": OrderStatus.CANCELED, "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _fmt(x: float) -> str:
    return f"{x:.2f}" if x >= 1 else f"{x:.4f}"


class AlpacaBroker:
    name = "alpaca"

    def __init__(self, key_id: str, secret_key: str, live: bool = False, data_feed: str = "iex",
                 transport: httpx2.BaseTransport | None = None, timeout: float = 15.0):
        if not key_id or not secret_key:
            raise BrokerError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY")
        headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self.live = live
        self.data_feed = data_feed
        self._trading = httpx2.Client(base_url=LIVE_URL if live else PAPER_URL, headers=headers,
                                      timeout=timeout, transport=transport)
        self._data = httpx2.Client(base_url=DATA_URL, headers=headers, timeout=timeout, transport=transport)

    # --- http -------------------------------------------------------------------------

    def _request(self, client: httpx2.Client, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = client.request(method, path, **kwargs)
        except httpx2.HTTPError as e:
            raise BrokerError(f"Alpaca {method} {path} failed: {e}") from e
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text)
            except ValueError:
                message = resp.text
            raise BrokerError(f"Alpaca {method} {path} -> {resp.status_code}: {message}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # --- broker protocol --------------------------------------------------------------

    def clock(self) -> Clock:
        c = self._request(self._trading, "GET", "/v2/clock")
        return Clock(now=_ts(c["timestamp"]), is_open=bool(c["is_open"]),
                     next_open=_ts(c.get("next_open")), next_close=_ts(c.get("next_close")))

    def get_account(self) -> Account:
        a = self._request(self._trading, "GET", "/v2/account")
        equity = float(a["equity"])
        return Account(equity=equity, cash=float(a["cash"]), buying_power=float(a["buying_power"]),
                       day_start_equity=float(a.get("last_equity") or equity),
                       currency=a.get("currency", "USD"))

    def get_positions(self) -> dict[str, Position]:
        rows = self._request(self._trading, "GET", "/v2/positions")
        return {
            r["symbol"]: Position(symbol=r["symbol"], qty=float(r["qty"]),
                                  avg_price=float(r["avg_entry_price"]),
                                  market_price=float(r.get("current_price") or r["avg_entry_price"]))
            for r in rows
        }

    def _raw_open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"status": "open", "limit": 500}
        if symbol:
            params["symbols"] = symbol
        return self._request(self._trading, "GET", "/v2/orders", params=params)

    def get_open_orders(self) -> list[Order]:
        return [self._parse_order(o) for o in self._raw_open_orders() if o["type"] not in _PROTECTIVE_TYPES]

    def snapshot(self) -> tuple[Account, dict[str, Position], list[Order]]:
        # Not atomic over REST. Open orders first: an entry that fills mid-read is counted twice
        # (open order and position) rather than not at all.
        open_orders = self.get_open_orders()
        positions = self.get_positions()
        return self.get_account(), positions, open_orders

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        # Calendar window wide enough to cover `limit` trading days, then keep the newest `limit`.
        start = datetime.now(timezone.utc) - timedelta(days=int(limit * 1.5) + 10)
        params: dict[str, Any] = {"timeframe": "1Day", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "adjustment": "all", "feed": self.data_feed, "limit": 10_000}
        bars: list[Bar] = []
        while True:
            page = self._request(self._data, "GET", f"/v2/stocks/{symbol}/bars", params=params)
            for b in page.get("bars") or []:
                bars.append(Bar(t=_ts(b["t"]), open=b["o"], high=b["h"], low=b["l"], close=b["c"], volume=b["v"]))
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return bars[-limit:]

    def get_latest_price(self, symbol: str) -> float:
        r = self._request(self._data, "GET", f"/v2/stocks/{symbol}/trades/latest", params={"feed": self.data_feed})
        return float(r["trade"]["p"])

    def submit_order(self, order: OrderRequest) -> Order:
        if order.side == Side.SELL:
            return self._submit_exit(order)
        return self._post_order(order)

    def _post_order(self, order: OrderRequest) -> Order:
        body: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": f"{order.qty:g}",
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": "day",
        }
        if order.order_type == OrderType.LIMIT and order.limit_price:
            body["limit_price"] = _fmt(order.limit_price)
        if order.side == Side.BUY and order.stop_loss_price:
            # OTO: the stop leg is created once the entry fills. GTC so the stop outlives the session.
            body["order_class"] = "oto"
            body["time_in_force"] = "gtc"
            body["stop_loss"] = {"stop_price": _fmt(order.stop_loss_price)}
        return self._parse_order(self._request(self._trading, "POST", "/v2/orders", json=body), order)

    def _submit_exit(self, order: OrderRequest) -> Order:
        """Protective stops reserve the shares they cover, so cancel them, sell, then re-protect the rest."""
        stops = [o for o in self._raw_open_orders(order.symbol)
                 if o["type"] in _PROTECTIVE_TYPES and o["side"] == "sell" and o["symbol"] == order.symbol]
        stop_price = min((float(o["stop_price"]) for o in stops if o.get("stop_price")), default=None)
        for o in stops:
            self._request(self._trading, "DELETE", f"/v2/orders/{o['id']}")
        held = self.get_positions().get(order.symbol)
        held_qty = held.qty if held else 0.0
        try:
            result = self._post_order(order)
        except BrokerError:
            self._place_stop(order.symbol, held_qty, stop_price)  # never leave the position unprotected
            raise
        self._place_stop(order.symbol, held_qty - order.qty, stop_price)
        return result

    def _place_stop(self, symbol: str, qty: float, stop_price: float | None) -> None:
        if stop_price is None or qty < 1:
            return
        self._request(self._trading, "POST", "/v2/orders", json={
            "symbol": symbol, "qty": f"{int(qty)}", "side": "sell",
            "type": "stop", "stop_price": _fmt(stop_price), "time_in_force": "gtc",
        })

    def cancel_all_orders(self) -> None:
        self._request(self._trading, "DELETE", "/v2/orders")

    def close_all_positions(self) -> None:
        self._request(self._trading, "DELETE", "/v2/positions", params={"cancel_orders": "true"})

    # --- parsing ----------------------------------------------------------------------

    @staticmethod
    def _parse_order(o: dict, request: OrderRequest | None = None) -> Order:
        if request is None:
            otype = OrderType.LIMIT if o.get("type") == "limit" else OrderType.MARKET
            request = OrderRequest(symbol=o["symbol"], side=Side(o["side"]), qty=float(o.get("qty") or 0),
                                   order_type=otype,
                                   limit_price=float(o["limit_price"]) if o.get("limit_price") else None)
        filled_avg = o.get("filled_avg_price")
        return Order(id=o["id"], request=request, status=_STATUS.get(o.get("status", ""), OrderStatus.PENDING),
                     submitted_at=_ts(o.get("submitted_at")) or datetime.now(timezone.utc),
                     filled_qty=float(o.get("filled_qty") or 0),
                     filled_avg_price=float(filled_avg) if filled_avg else None)
