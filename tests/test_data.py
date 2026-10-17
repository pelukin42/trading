from datetime import date

import pandas as pd
import pytest

from papertrade.config import DataConfig
from papertrade.data import DataError, MarketData, load_market_data


def frame(dates, opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=pd.DatetimeIndex(dates))


def test_from_frames_aligns_trims_and_fills():
    a = frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5])
    b = frame(["2024-01-02", "2024-01-04"], [None, 20], [10, 21])
    data = MarketData.from_frames({"A": a, "B": b})

    assert list(data.dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert data.close.loc["2024-01-03", "B"] == 10  # sin dato: se arrastra el último cierre
    assert data.open.loc["2024-01-02", "B"] == 10  # sin apertura: se usa el cierre
    assert data.until(pd.Timestamp("2024-01-03")).dates[-1] == pd.Timestamp("2024-01-03")


def test_from_frames_rejects_empty_symbol():
    with pytest.raises(DataError, match="B"):
        MarketData.from_frames({"A": frame(["2024-01-01"], [1], [1]), "B": frame([], [], [])})


def test_csv_loader_filters_dates(csv_dir):
    cfg = DataConfig(source="csv", csv_dir=str(csv_dir))
    data = load_market_data(cfg, ["AAA", "SPY"], date(2023, 3, 1), date(2023, 3, 31))
    assert data.dates[0] >= pd.Timestamp("2023-03-01")
    assert data.dates[-1] <= pd.Timestamp("2023-03-31")
    assert list(data.close.columns) == ["AAA", "SPY"]


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.params = []

    def get(self, url, params, headers, timeout):
        self.params.append(dict(params))
        return FakeResponse(self.pages[len(self.params) - 1])


def test_alpaca_loader_paginates_and_uses_new_york_dates(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    pages = [
        {"bars": {"AAPL": [{"t": "2024-01-02T05:00:00Z", "o": 1, "c": 2}]}, "next_page_token": "p2"},
        {"bars": {"SPY": [{"t": "2024-01-02T05:00:00Z", "o": 3, "c": 4}]}, "next_page_token": None},
    ]
    session = FakeSession(pages)
    cfg = DataConfig(source="alpaca", feed="iex")
    data = load_market_data(cfg, ["AAPL", "SPY"], date(2024, 1, 1), adjust="split", session=session)

    assert data.dates[0] == pd.Timestamp("2024-01-02")
    assert data.close.loc["2024-01-02", "SPY"] == 4
    assert session.params[0]["adjustment"] == "split" and session.params[0]["feed"] == "iex"
    assert session.params[1]["page_token"] == "p2"


def test_alpaca_loader_requires_keys(monkeypatch):
    for var in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(DataError, match="ALPACA_API_KEY"):
        load_market_data(DataConfig(source="alpaca"), ["SPY"], date(2024, 1, 1))
