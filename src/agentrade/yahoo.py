"""Daily bars, delayed quotes and splits/dividends from Yahoo Finance's public chart endpoint
(no account or key needed).

Unofficial and meant for personal use and testing: Yahoo's terms don't allow redistributing its data,
so a product sold to others needs a licensed market data feed.

Prices are split-adjusted after the fact: once a stock splits, Yahoo restates its whole history in
post-split shares. Callers holding shares must apply the split events to stay in the same units.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx2

from . import market_calendar as cal
from .broker.base import BrokerError
from .models import Bar, CorporateAction

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Smallest Yahoo range that holds `n` trading days.
_RANGES = [(5, "5d"), (21, "1mo"), (63, "3mo"), (126, "6mo"), (252, "1y"), (504, "2y"), (1260, "5y")]
# Tickers such as AAPL, BRK-B, BF.B, ^GSPC or EURUSD=X.
_SYMBOL = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


@dataclass(frozen=True)
class Quote:
    price: float
    t: datetime


def range_for(limit: int) -> str:
    for n, r in _RANGES:
        if limit <= n:
            return r
    return "10y"


def clean_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not _SYMBOL.match(s):
        raise BrokerError(f"Invalid ticker symbol: {symbol!r}")
    return s


class YahooData:
    def __init__(self, transport: httpx2.BaseTransport | None = None, timeout: float = 15.0,
                 cache_seconds: float = 60.0):
        self._client = httpx2.Client(timeout=timeout, transport=transport,
                                     headers={"User-Agent": "Mozilla/5.0 (agentrade)"})
        self.cache_seconds = cache_seconds
        # Values are a chart result, or the message of a "no such symbol" answer (also worth caching).
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any] | str]] = {}

    def _chart(self, symbol: str, range_: str) -> dict[str, Any]:
        symbol = clean_symbol(symbol)
        key = (symbol, range_)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_seconds:
            if isinstance(hit[1], str):
                raise BrokerError(hit[1])
            return hit[1]
        try:
            resp = self._client.get(CHART_URL.format(symbol=quote(symbol, safe="^")),
                                    params={"range": range_, "interval": "1d", "events": "div,split"})
        except (httpx2.HTTPError, httpx2.InvalidURL) as e:
            raise BrokerError(f"Yahoo Finance unreachable for {symbol}: {e}") from e
        if resp.status_code == 429:
            raise BrokerError("Yahoo Finance rate limit hit; retry in a minute")
        try:
            chart = resp.json()["chart"]
        except (ValueError, KeyError) as e:
            raise BrokerError(f"Unexpected Yahoo Finance response for {symbol} ({resp.status_code})") from e
        if chart.get("error") or not chart.get("result"):
            message = f"Yahoo Finance: {symbol}: {(chart.get('error') or {}).get('description', 'no data')}"
            self._cache[key] = (time.monotonic(), message)
            raise BrokerError(message)
        result = chart["result"][0]
        self._cache[key] = (time.monotonic(), result)
        return result

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        """Daily bars, oldest first. During a session the last bar is today's, still in progress."""
        result = self._chart(symbol, range_for(limit))
        quote_ = result["indicators"]["quote"][0]
        bars = []
        for i, ts in enumerate(result.get("timestamp") or []):
            o, h, l, c = (quote_[k][i] for k in ("open", "high", "low", "close"))
            if None in (o, h, l, c):
                continue  # Yahoo leaves gaps as nulls
            bars.append(Bar(datetime.fromtimestamp(ts, timezone.utc), o, h, l, c, quote_["volume"][i] or 0.0))
        return bars[-limit:]

    def get_quote(self, symbol: str) -> Quote:
        meta = self._chart(symbol, "5d")["meta"]
        price = meta.get("regularMarketPrice")
        if not price:
            raise BrokerError(f"Yahoo Finance: no price for {symbol}")
        return Quote(float(price), datetime.fromtimestamp(meta["regularMarketTime"], timezone.utc))

    def get_corporate_actions(self, symbol: str, since: date) -> list[CorporateAction]:
        """Splits and dividends effective after `since`, oldest first."""
        days = (datetime.now(timezone.utc).date() - since).days
        events = self._chart(symbol, range_for(max(days, 5) + 5)).get("events") or {}
        out = []
        for e in (events.get("splits") or {}).values():
            if e.get("numerator") and e.get("denominator"):
                out.append(CorporateAction(cal.session_date(datetime.fromtimestamp(e["date"], timezone.utc)),
                                           "split", ratio=e["numerator"] / e["denominator"]))
        for e in (events.get("dividends") or {}).values():
            if e.get("amount"):
                out.append(CorporateAction(cal.session_date(datetime.fromtimestamp(e["date"], timezone.utc)),
                                           "dividend", amount=float(e["amount"])))
        return sorted((a for a in out if a.session > since), key=lambda a: (a.session, a.kind != "split"))
