from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from agentrade.broker.base import Clock
from agentrade.config import LIVE_CONFIRMATION, Settings, load_dotenv
from agentrade.engine import build_stack, run_loop
from agentrade.journal import Journal


class ClockBroker:
    name = "fake"

    def __init__(self, is_open):
        self.is_open = is_open

    def clock(self):
        now = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)
        return Clock(now=now, is_open=self.is_open, next_open=now + timedelta(hours=18))


class CountingAgent:
    def __init__(self):
        self.cycles = 0

    def run_cycle(self):
        self.cycles += 1
        return NS(status="completed")


def test_live_mode_requires_typed_confirmation(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTRADE_LIVE_CONFIRM", raising=False)
    s = Settings(mode="live", data_dir=tmp_path)
    with pytest.raises(PermissionError):
        s.validate_live()
    monkeypatch.setenv("AGENTRADE_LIVE_CONFIRM", LIVE_CONFIRMATION)
    s.validate_live()


def test_dotenv_does_not_override_existing_env(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("AGENTRADE_WATCHLIST=aaa, bbb\nAGENTRADE_EFFORT=high\n# comment\n")
    monkeypatch.setenv("AGENTRADE_EFFORT", "low")
    monkeypatch.delenv("AGENTRADE_WATCHLIST", raising=False)
    load_dotenv(env)
    s = Settings.from_env()
    assert s.watchlist == ["AAA", "BBB"] and s.effort == "low"


def test_run_loop_respects_kill_switch_and_daily_cap(tmp_path):
    settings = Settings(data_dir=tmp_path, max_cycles_per_day=1)
    stack = build_stack(settings, broker=ClockBroker(is_open=True), journal=Journal(":memory:"))
    agent = CountingAgent()

    settings.kill_switch_path.write_text("x")
    run_loop(stack, agent, once=True)
    assert agent.cycles == 0

    settings.kill_switch_path.unlink()
    run_loop(stack, agent, once=True)
    assert agent.cycles == 1

    stack.journal.start_cycle("m", "paper", t=stack.broker.clock().now)
    run_loop(stack, agent, once=True)
    assert agent.cycles == 1  # daily cap reached


def test_run_loop_survives_broker_outage(tmp_path):
    from agentrade.broker.base import BrokerError

    class FlakyBroker(ClockBroker):
        calls = 0

        def clock(self):
            FlakyBroker.calls += 1
            if FlakyBroker.calls == 1:
                raise BrokerError("503")
            return super().clock()

    settings = Settings(data_dir=tmp_path)
    stack = build_stack(settings, broker=FlakyBroker(is_open=True), journal=Journal(":memory:"))
    agent = CountingAgent()
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if agent.cycles:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_loop(stack, agent, sleep=fake_sleep)
    assert sleeps[0] == 60 and agent.cycles == 1


def test_run_loop_waits_for_market_open(tmp_path):
    settings = Settings(data_dir=tmp_path)
    stack = build_stack(settings, broker=ClockBroker(is_open=False), journal=Journal(":memory:"))
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_loop(stack, CountingAgent(), sleep=fake_sleep)
    assert sleeps == [3600]  # capped at an hour, then re-checks the clock
