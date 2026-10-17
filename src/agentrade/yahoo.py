"""Daily bars and delayed quotes from Yahoo Finance's public chart endpoint (no account or key needed).

Unofficial and meant for personal use and testing: Yahoo's terms don't allow redistributing its data,
so a product sold to others needs a licensed market data feed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx2

from .broker.base import BrokerError
from .models import Bar

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Smallest Yahoo range that holds `n` trading days.
_RANGES = [(5, "5d"), (21, "1mo"), (63, "3mo"), (126, "6mo"), (252, "1y"), (504, "2y"), (1260, "5y")]


@dataclass(frozen=True)
class Quote:
    price: float
    t: datetime


def range_for(limit: int) -> str:
    for n, r in _RANGES:
        if limit <= n:
            return r
    return "10y"


class YahooData:
    def __init__(self, transport: httpx2.BaseTransport | None = None, timeout: float = 15.0,
                 cache_seconds: float = 60.0):
        self._client = httpx2.Client(timeout=timeout, transport=transport,
                                     headers={"User-Agent": "Mozilla/5.0 (agentrade)"})
        self.cache_seconds = cache_seconds
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def _chart(self, symbol: str, range_: str) -> dict[str, Any]:
        key = (symbol, range_)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        try:
            resp = self._client.get(CHART_URL.format(symbol=symbol), params={"range": range_, "interval": "1d"})
        except httpx2.HTTPError as e:
            raise BrokerError(f"Yahoo Finance unreachable for {symbol}: {e}") from e
        if resp.status_code == 429:
            raise BrokerError("Yahoo Finance rate limit hit; retry in a minute")
        try:
            chart = resp.json()["chart"]
        except (ValueError, KeyError) as e:
            raise BrokerError(f"Unexpected Yahoo Finance response for {symbol} ({resp.status_code})") from e
        if chart.get("error") or not chart.get("result"):
            desc = (chart.get("error") or {}).get("description", "no data")
            raise BrokerError(f"Yahoo Finance: {symbol}: {desc}")
        result = chart["result"][0]
        self._cache[key] = (time.monotonic(), result)
        return result

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        """Daily bars, oldest first. During a session the last bar is today's, still in progress."""
        result = self._chart(symbol.upper(), range_for(limit))
        quote = result["indicators"]["quote"][0]
        bars = []
        for i, ts in enumerate(result.get("timestamp") or []):
            o, h, l, c = (quote[k][i] for k in ("open", "high", "low", "close"))
            if None in (o, h, l, c):
                continue  # Yahoo leaves gaps as nulls
            bars.append(Bar(datetime.fromtimestamp(ts, timezone.utc), o, h, l, c, quote["volume"][i] or 0.0))
        return bars[-limit:]

    def get_quote(self, symbol: str) -> Quote:
        meta = self._chart(symbol.upper(), "5d")["meta"]
        price = meta.get("regularMarketPrice")
        if not price:
            raise BrokerError(f"Yahoo Finance: no price for {symbol}")
        return Quote(float(price), datetime.fromtimestamp(meta["regularMarketTime"], timezone.utc))
