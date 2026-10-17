import httpx2
import pytest

from agentrade.broker.base import BrokerError
from agentrade.yahoo import YahooData, range_for


def chart(timestamps, opens, closes, price=101.5, market_time=1792180800):
    n = len(timestamps)
    return {"chart": {"error": None, "result": [{
        "meta": {"regularMarketPrice": price, "regularMarketTime": market_time},
        "timestamp": timestamps,
        "indicators": {"quote": [{"open": opens, "high": [c and c + 1 for c in closes],
                                  "low": [c and c - 1 for c in closes], "close": closes,
                                  "volume": [1000] * n}]},
    }]}}


class FakeYahoo:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if "NOPE" in request.url.path:
            return httpx2.Response(404, json={"chart": {"result": None, "error": {
                "code": "Not Found", "description": "No data found, symbol may be delisted"}}})
        return httpx2.Response(200, json=chart([1791811800, 1791898200, 1791984600],
                                               [100.0, None, 102.0], [100.5, None, 101.5]))


@pytest.fixture
def fake():
    return FakeYahoo()


@pytest.fixture
def yahoo(fake):
    return YahooData(transport=httpx2.MockTransport(fake))


def test_bars_skip_null_rows_and_request_the_right_range(yahoo, fake):
    bars = yahoo.get_bars("spy", limit=30)
    assert [b.close for b in bars] == [100.5, 101.5]
    assert bars[0].t.tzinfo is not None
    req = fake.requests[0]
    assert req.url.path == "/v8/finance/chart/SPY"
    assert req.url.params["range"] == "3mo" and req.url.params["interval"] == "1d"


def test_quote_and_cache(yahoo, fake):
    q = yahoo.get_quote("SPY")
    assert q.price == 101.5 and q.t.year == 2026
    yahoo.get_quote("SPY")
    yahoo.get_bars("SPY", limit=5)  # same 5d range: served from cache
    assert len(fake.requests) == 1


def test_unknown_symbol_raises_broker_error(yahoo):
    with pytest.raises(BrokerError, match="delisted"):
        yahoo.get_bars("NOPE", 10)


def test_range_mapping():
    assert [range_for(n) for n in (5, 20, 60, 120, 252, 260, 1000, 5000)] == \
        ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
