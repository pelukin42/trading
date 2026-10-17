import pytest

from papertrade.alpaca import PAPER_URL, AlpacaPaperBroker, BrokerError
from papertrade.engine import Order


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.text = str(payload)
        self.content = b"x" if payload is not None else b""

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.routes[(method, url.removeprefix(PAPER_URL))]


def test_live_url_is_blocked():
    with pytest.raises(BrokerError, match="paper"):
        AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")


def test_account_and_submit():
    session = FakeSession(
        {
            ("GET", "/v2/account"): FakeResponse({"cash": "1500.5", "equity": "10000"}),
            ("GET", "/v2/positions"): FakeResponse([{"symbol": "AAPL", "qty": "3", "current_price": "200.1"}]),
            ("POST", "/v2/orders"): FakeResponse({"id": "abc", "status": "accepted"}),
            ("DELETE", "/v2/orders"): FakeResponse(None, status=207),
        }
    )
    broker = AlpacaPaperBroker("key", "secret", session=session)

    snap = broker.account()
    assert snap.cash == 1500.5 and snap.equity == 10000
    assert snap.positions == {"AAPL": 3.0} and snap.prices == {"AAPL": 200.1}

    broker.cancel_open_orders()
    resp = broker.submit(Order("MSFT", "buy", 2.5))
    assert resp["id"] == "abc"
    method, url, kwargs = session.calls[-1]
    assert url == f"{PAPER_URL}/v2/orders"
    assert kwargs["json"] == {"symbol": "MSFT", "qty": "2.5", "side": "buy", "type": "market", "time_in_force": "day"}
    assert kwargs["headers"]["APCA-API-KEY-ID"] == "key"


def test_http_errors_raise():
    session = FakeSession({("GET", "/v2/account"): FakeResponse({"message": "forbidden"}, status=403)})
    with pytest.raises(BrokerError, match="403"):
        AlpacaPaperBroker("k", "s", session=session).account()


def test_full_liquidation_closes_position_instead_of_rounding_qty():
    session = FakeSession({("DELETE", "/v2/positions/AAPL"): FakeResponse({"id": "c1", "status": "accepted"})})
    resp = AlpacaPaperBroker("k", "s", session=session).submit(Order("AAPL", "sell", 1.2345678, close_position=True))
    assert resp["id"] == "c1"
    assert session.calls[0][0] == "DELETE"
