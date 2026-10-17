"""NYSE regular-session calendar (9:30-16:00 ET), with official holidays and early closes.

Source: NYSE Group 2026-2028 holiday calendar (nyse.com/markets/hours-calendars, ICE press release
2025-12-23). Years not listed below fall back to weekdays only; extend the tables when NYSE publishes them.
Unscheduled closures (e.g. national days of mourning) are not known in advance.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .broker.base import Clock

ET = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)

HOLIDAYS: dict[date, str] = {
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "Martin Luther King, Jr. Day",
    date(2026, 2, 16): "Washington's Birthday",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (observed)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving Day",
    date(2026, 12, 25): "Christmas Day",
    date(2027, 1, 1): "New Year's Day",
    date(2027, 1, 18): "Martin Luther King, Jr. Day",
    date(2027, 2, 15): "Washington's Birthday",
    date(2027, 3, 26): "Good Friday",
    date(2027, 5, 31): "Memorial Day",
    date(2027, 6, 18): "Juneteenth (observed)",
    date(2027, 7, 5): "Independence Day (observed)",
    date(2027, 9, 6): "Labor Day",
    date(2027, 11, 25): "Thanksgiving Day",
    date(2027, 12, 24): "Christmas Day (observed)",
}

EARLY_CLOSES: dict[date, time] = {
    date(2026, 11, 27): time(13, 0),
    date(2026, 12, 24): time(13, 0),
    date(2027, 11, 26): time(13, 0),
}

COVERED_YEARS = {2026, 2027}

log = logging.getLogger("agentrade")
_warned_years: set[int] = set()


def is_trading_day(d: date) -> bool:
    if d.year not in COVERED_YEARS and d.year not in _warned_years:
        _warned_years.add(d.year)
        log.warning("No NYSE holiday table for %d; treating every weekday as a trading day.", d.year)
    return d.weekday() < 5 and d not in HOLIDAYS


def session_bounds(d: date) -> tuple[datetime, datetime]:
    """Open and close of the regular session on trading day `d`, in UTC."""
    close = EARLY_CLOSES.get(d, CLOSE)
    return (datetime.combine(d, OPEN, ET).astimezone(timezone.utc),
            datetime.combine(d, close, ET).astimezone(timezone.utc))


def session_date(t: datetime) -> date:
    """Calendar date in New York of a timestamp (e.g. a daily bar's time)."""
    return t.astimezone(ET).date()


def last_session_started(now: datetime) -> date:
    """Most recent trading day whose session has opened at or before `now` (it may still be in progress)."""
    d = session_date(now)
    while True:
        if is_trading_day(d) and session_bounds(d)[0] <= now:
            return d
        d -= timedelta(days=1)


def next_session_after(now: datetime) -> date:
    """First trading day whose session opens strictly after `now`."""
    d = session_date(now)
    while True:
        if is_trading_day(d) and session_bounds(d)[0] > now:
            return d
        d += timedelta(days=1)


def is_open(now: datetime) -> bool:
    d = session_date(now)
    if not is_trading_day(d):
        return False
    start, end = session_bounds(d)
    return start <= now < end


def clock(now: datetime) -> Clock:
    open_now = is_open(now)
    next_open = session_bounds(next_session_after(now))[0]
    current = last_session_started(now)
    next_close = session_bounds(current)[1] if open_now else session_bounds(next_session_after(now))[1]
    return Clock(now=now, is_open=open_now, next_open=next_open, next_close=next_close)
