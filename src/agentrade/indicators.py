"""Technical indicators in plain Python. Each returns the latest value, or None if history is too short."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models import Bar

TRADING_DAYS = 252


def sma(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    current = sum(values[:period]) / period
    for v in values[period:]:
        current = v * k + current * (1 - k)
    return current


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI."""
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(values[:-1], values[1:]):
        delta = cur - prev
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Wilder's Average True Range."""
    if len(bars) < period + 1:
        return None
    trs = []
    for prev, cur in zip(bars[:-1], bars[1:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    current = sum(trs[:period]) / period
    for tr in trs[period:]:
        current = (current * (period - 1) + tr) / period
    return current


def pct_return(values: Sequence[float], lookback: int) -> float | None:
    if len(values) <= lookback or values[-lookback - 1] == 0:
        return None
    return values[-1] / values[-lookback - 1] - 1


def log_returns(values: Sequence[float]) -> list[float]:
    return [math.log(b / a) for a, b in zip(values[:-1], values[1:]) if a > 0 and b > 0]


def annualized_volatility(values: Sequence[float], lookback: int = 20,
                          periods_per_year: int = TRADING_DAYS) -> float | None:
    rets = log_returns(values[-lookback - 1:])
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)
