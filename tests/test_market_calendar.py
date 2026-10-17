from datetime import date, datetime, timezone

from agentrade import market_calendar as cal


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_weekend_is_closed_and_next_open_is_monday():
    c = cal.clock(utc(2026, 10, 17, 15))  # Saturday
    assert not c.is_open
    assert c.next_open == utc(2026, 10, 19, 13, 30)  # 9:30 EDT


def test_regular_session_hours():
    assert not cal.is_open(utc(2026, 10, 19, 13, 29))
    assert cal.is_open(utc(2026, 10, 19, 13, 30))
    assert cal.is_open(utc(2026, 10, 19, 19, 59))
    assert not cal.is_open(utc(2026, 10, 19, 20, 0))
    assert cal.clock(utc(2026, 10, 19, 15)).next_close == utc(2026, 10, 19, 20)


def test_daylight_saving_shift_moves_the_open_in_utc():
    # DST ends 2026-11-01: the open moves from 13:30 to 14:30 UTC.
    assert not cal.is_open(utc(2026, 11, 2, 14, 0))
    assert cal.is_open(utc(2026, 11, 2, 14, 31))


def test_holidays_and_early_closes():
    assert not cal.is_open(utc(2026, 11, 26, 16))      # Thanksgiving
    assert cal.is_open(utc(2026, 11, 27, 17, 59))      # day after: closes 13:00 EST = 18:00 UTC
    assert not cal.is_open(utc(2026, 11, 27, 18, 0))
    assert not cal.is_trading_day(date(2027, 12, 24))  # Christmas observed
    assert cal.is_trading_day(date(2027, 12, 31))      # New Year's 2028 falls on a Saturday: not observed


def test_session_helpers_skip_holidays():
    assert cal.next_session_after(utc(2026, 7, 2, 21)) == date(2026, 7, 6)  # July 3 observed holiday
    assert cal.last_session_started(utc(2026, 10, 17, 15)) == date(2026, 10, 16)
    assert cal.last_session_started(utc(2026, 10, 19, 12)) == date(2026, 10, 16)  # Monday before the open
    assert cal.last_session_started(utc(2026, 10, 19, 14)) == date(2026, 10, 19)


def test_years_without_a_table_fall_back_to_weekdays():
    assert cal.is_trading_day(date(2030, 1, 1))
    assert not cal.is_trading_day(date(2030, 1, 5))  # Saturday
