"""Wiring for live/paper operation and the scheduling loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic

from .agent import CycleResult, TradingAgent
from .broker.alpaca import AlpacaBroker
from .broker.base import Broker, BrokerError
from .broker.local_paper import LocalPaperBroker
from .config import Settings
from .journal import Journal
from .risk import RiskManager
from .router import OrderRouter
from .yahoo import YahooData

log = logging.getLogger("agentrade")


@dataclass
class Stack:
    settings: Settings
    broker: Broker
    journal: Journal
    risk: RiskManager
    router: OrderRouter


def make_broker(settings: Settings) -> Broker:
    if settings.broker == "alpaca":
        return AlpacaBroker(settings.alpaca_key_id, settings.alpaca_secret_key,
                            live=settings.mode == "live", data_feed=settings.alpaca_data_feed)
    return LocalPaperBroker(YahooData(), settings.paper_state_path, initial_cash=settings.paper_initial_cash)


def build_stack(settings: Settings, broker: Broker | None = None, journal: Journal | None = None) -> Stack:
    settings.validate_live()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    broker = broker or make_broker(settings)
    journal = journal or Journal(settings.journal_path)
    risk = RiskManager(settings.risk, allowed_symbols=settings.watchlist, kill_switch_path=settings.kill_switch_path)
    return Stack(settings, broker, journal, risk, OrderRouter(broker, risk, journal))


def build_agent(stack: Stack, client: anthropic.Anthropic | None = None) -> TradingAgent:
    return TradingAgent(client or anthropic.Anthropic(), stack.broker, stack.router, stack.journal, stack.settings)


def _midnight_utc(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def run_loop(stack: Stack, agent: TradingAgent, once: bool = False,
             on_cycle=lambda result: None, sleep=time.sleep) -> None:
    """Run cycles while the market is open, every `cycle_minutes`, capped at `max_cycles_per_day`."""
    s = stack.settings
    while True:
        if stack.risk.kill_switch_active():
            log.warning("Kill switch active; not trading. Run `agentrade resume` to continue.")
            if once:
                return
            sleep(60)
            continue

        try:
            clock = stack.broker.clock()
        except BrokerError as e:
            log.error("Broker unavailable: %s", e)
            if once:
                raise
            sleep(60)
            continue
        if not clock.is_open:
            if once:
                log.warning("Market is closed: orders from this cycle will queue for the next open (%s).",
                            clock.next_open)
            else:
                wait = (clock.next_open - clock.now).total_seconds() if clock.next_open else 3600
                log.info("Market closed; next open %s", clock.next_open)
                sleep(min(max(wait, 60), 3600))
                continue

        if stack.journal.cycles_since(_midnight_utc(clock.now)) >= s.max_cycles_per_day:
            log.info("Reached max_cycles_per_day=%d", s.max_cycles_per_day)
            if once:
                return
            sleep(3600)
            continue

        result: CycleResult = agent.run_cycle()
        on_cycle(result)
        if once:
            return
        sleep(s.cycle_minutes * 60)
