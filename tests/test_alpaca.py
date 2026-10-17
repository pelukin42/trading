"""AlpacaBroker request/response handling against a mocked HTTP transport."""

import json

import httpx2
import pytest

from agentrade.broker.alpaca import AlpacaBroker
from agentrade.broker.base import BrokerError
from agentrade.models import OrderRequest, OrderType, Side


class FakeAlpaca:
    def __init__(self):
        self.requests = []
        self.open_orders = []
        self.positions = [{"symbol": "AAPL", "qty": "30", "avg_entry_price": "150", "current_price": "160"}]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, dict(request.url.params), body))
        path = request.url.path
        if path == "/v2/account":
            return httpx2.Response(200, json={"equity": "100000", "cash": "40000", "buying_power": "80000",
                                              "last_equity": "101000", "currency": "USD"})
        if path == "/v2/positions" and request.method == "GET":
            return httpx2.Response(200, json=self.positions)
        if path == "/v2/orders" and request.method == "GET":
            return httpx2.Response(200, json=self.open_orders)
        if path == "/v2/orders" and request.method == "POST":
            if body["symbol"] == "BAD" and body["type"] != "stop":
                return httpx2.Response(403, json={"message": "insufficient buying power"})
            return httpx2.Response(200, json={"id": "ord-1", "symbol": body["symbol"], "side": body["side"],
                                              "qty": body["qty"], "type": body["type"], "status": "accepted",
                                              "filled_qty": "0", "submitted_at": "2026-10-16T14:00:00Z"})
        if path.startswith("/v2/orders/") and request.method == "DELETE":
            return httpx2.Response(204)
        if path == "/v2/stocks/AAPL/bars":
            if "page_token" not in request.url.params:
                return httpx2.Response(200, json={"bars": [{"t": "2026-10-14T04:00:00Z", "o": 1, "h": 2, "l": 0.5,
                                                            "c": 1.5, "v": 100}], "next_page_token": "p2"})
            return httpx2.Response(200, json={"bars": [{"t": "2026-10-15T04:00:00Z", "o": 1.5, "h": 2.5, "l": 1,
                                                        "c": 2, "v": 200}], "next_page_token": None})
        if path == "/v2/stocks/AAPL/trades/latest":
            return httpx2.Response(200, json={"symbol": "AAPL", "trade": {"p": 161.25}})
        if path == "/v2/clock":
            return httpx2.Response(200, json={"timestamp": "2026-10-16T14:00:00Z", "is_open": True,
                                              "next_open": "2026-10-17T13:30:00Z",
                                              "next_close": "2026-10-16T20:00:00Z"})
        return httpx2.Response(404, json={"message": f"unexpected {request.method} {path}"})


@pytest.fixture
def fake():
    return FakeAlpaca()


@pytest.fixture
def broker(fake):
    return AlpacaBroker("key", "secret", transport=httpx2.MockTransport(fake))


def test_account_positions_clock_and_price(broker):
    a = broker.get_account()
    assert (a.equity, a.cash, a.day_start_equity) == (100_000, 40_000, 101_000)
    assert broker.get_positions()["AAPL"].qty == 30
    assert broker.clock().is_open
    assert broker.get_latest_price("AAPL") == 161.25


def test_bars_follow_pagination(broker):
    bars = broker.get_bars("AAPL", limit=10)
    assert [b.close for b in bars] == [1.5, 2]


def test_buy_with_stop_is_sent_as_gtc_oto(broker, fake):
    order = broker.submit_order(OrderRequest("AAPL", Side.BUY, 10, stop_loss_price=150.0))
    _, _, _, body = fake.requests[-1]
    assert body == {"symbol": "AAPL", "qty": "10", "side": "buy", "type": "market", "time_in_force": "gtc",
                    "order_class": "oto", "stop_loss": {"stop_price": "150.00"}}
    assert order.id == "ord-1"


def test_limit_order_body(broker, fake):
    broker.submit_order(OrderRequest("AAPL", Side.BUY, 5, order_type=OrderType.LIMIT, limit_price=155.5))
    body = fake.requests[-1][3]
    assert body["type"] == "limit" and body["limit_price"] == "155.50" and body["time_in_force"] == "day"


def test_partial_exit_cancels_stop_then_reprotects_remaining_shares(broker, fake):
    fake.open_orders = [{"id": "stop-1", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": "30",
                         "stop_price": "140", "status": "new"}]
    broker.submit_order(OrderRequest("AAPL", Side.SELL, 10))
    calls = [(m, p, b) for m, p, _, b in fake.requests]
    assert ("DELETE", "/v2/orders/stop-1", None) in calls
    posts = [b for m, p, b in calls if m == "POST"]
    assert posts[0]["side"] == "sell" and posts[0]["qty"] == "10"
    assert posts[1] == {"symbol": "AAPL", "qty": "20", "side": "sell", "type": "stop", "stop_price": "140.00",
                        "time_in_force": "gtc"}


def test_rejected_exit_restores_the_stop_for_the_full_position(broker, fake):
    fake.positions = [{"symbol": "BAD", "qty": "30", "avg_entry_price": "10", "current_price": "10"}]
    fake.open_orders = [{"id": "stop-9", "symbol": "BAD", "side": "sell", "type": "stop", "qty": "30",
                         "stop_price": "9", "status": "new"}]
    with pytest.raises(BrokerError):
        broker.submit_order(OrderRequest("BAD", Side.SELL, 10))
    last = fake.requests[-1]
    assert last[0] == "POST" and last[3]["type"] == "stop" and last[3]["qty"] == "30"


def test_protective_stops_are_not_reported_as_open_orders(broker, fake):
    fake.open_orders = [
        {"id": "s", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": "30", "status": "new"},
        {"id": "l", "symbol": "AAPL", "side": "buy", "type": "limit", "qty": "5", "limit_price": "150",
         "status": "new", "filled_qty": "0"},
    ]
    assert [o.id for o in broker.get_open_orders()] == ["l"]


def test_api_errors_surface_alpacas_message(broker):
    with pytest.raises(BrokerError, match="insufficient buying power"):
        broker.submit_order(OrderRequest("BAD", Side.BUY, 1))


def test_paper_is_default_host(fake):
    b = AlpacaBroker("k", "s", transport=httpx2.MockTransport(fake))
    assert str(b._trading.base_url).startswith("https://paper-api.alpaca.markets")
