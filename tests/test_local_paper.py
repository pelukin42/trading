"""LocalPaperBroker against a fake data feed and a controllable clock."""

from datetime import date, datetime, timezone

import pytest

from agentrade.broker.local_paper import LocalPaperBroker
from agentrade.config import RiskLimits
from agentrade.journal import Journal
from agentrade.models import Bar, OrderRequest, OrderStatus, OrderType, Side
from agentrade.risk import RiskManager
from agentrade.router import OrderRouter
from agentrade.yahoo import Quote

SLIP = 5 / 10_000


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeData:
    """Daily bars stamped at the 9:30 ET open (13:30 UTC in October), visible once that time has passed."""

    def __init__(self, clock):
        self.clock = clock
        self.bars: dict[str, list[Bar]] = {}
        self.quotes: dict[str, float] = {}

    def add_bar(self, symbol, d: date, o, h, l, c):
        self.bars.setdefault(symbol, []).append(Bar(datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc),
                                                    o, h, l, c, 1e6))

    def get_bars(self, symbol, limit=200):
        return [b for b in self.bars.get(symbol, []) if b.t <= self.clock()][-limit:]

    def get_quote(self, symbol):
        return Quote(self.quotes[symbol], self.clock())


MON, TUE, WED = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)


@pytest.fixture
def env(tmp_path):
    clock = Clock(utc(2026, 10, 19, 15))  # Monday 11:00 ET, market open
    data = FakeData(clock)
    data.add_bar("AAA", date(2026, 10, 16), 99, 101, 98, 100)
    broker = LocalPaperBroker(data, tmp_path / "paper.json", initial_cash=100_000, now=clock)
    return clock, data, broker, tmp_path


def buy(qty=10, stop=None, **kw):
    return OrderRequest("AAA", Side.BUY, qty, stop_loss_price=stop, **kw)


def test_market_order_fills_now_while_open_and_state_persists(env):
    clock, data, broker, tmp_path = env
    data.quotes["AAA"] = 100.0
    o = broker.submit_order(buy(10))
    assert o.status == OrderStatus.FILLED and o.filled_avg_price == pytest.approx(100 * (1 + SLIP))
    cost = 10 * 100 * (1 + SLIP)
    assert broker.get_account().cash == pytest.approx(100_000 - cost * 1.0001)

    reopened = LocalPaperBroker(data, tmp_path / "paper.json", now=clock)
    assert reopened.get_positions()["AAA"].qty == 10


def test_order_placed_while_closed_fills_at_next_open(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)  # Saturday
    o = broker.submit_order(buy(10))
    assert o.status == OrderStatus.PENDING
    clock.t = utc(2026, 10, 19, 13)  # Monday before the open
    assert len(broker.get_open_orders()) == 1 and not broker.get_positions()
    clock.t = utc(2026, 10, 19, 14)
    data.add_bar("AAA", MON, 105, 106, 104, 105.5)
    data.quotes["AAA"] = 105.5
    assert broker.get_positions()["AAA"].avg_price == pytest.approx(105 * (1 + SLIP))
    assert broker.get_open_orders() == []


def test_market_order_waits_if_the_session_bar_is_late(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(buy(10))
    clock.t = utc(2026, 10, 19, 13, 31)  # open, but the feed has no Monday bar yet
    assert len(broker.get_open_orders()) == 1


def test_stop_triggers_on_a_live_quote(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10, stop=95))
    data.quotes["AAA"] = 94.0
    assert "AAA" not in broker.get_positions()
    fill = broker.get_fills()[-1]
    assert fill.reason == "stop" and fill.price == pytest.approx(94 * (1 - SLIP))


def test_entry_day_bar_is_ignored_for_intraday_entries_but_gaps_trigger_later(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10, stop=95))
    data.add_bar("AAA", MON, 99, 101, 90, 100)  # dipped to 90, possibly before our entry
    clock.t = utc(2026, 10, 19, 21)             # after the close
    assert "AAA" in broker.get_positions()
    data.add_bar("AAA", TUE, 93, 94, 92, 93.5)  # gapped below the stop
    clock.t = utc(2026, 10, 21, 12)             # Wednesday before the open
    data.quotes["AAA"] = 93.5
    assert "AAA" not in broker.get_positions()
    assert broker.get_fills()[-1].price == pytest.approx(93 * (1 - SLIP))


def test_open_fill_counts_the_entry_day_bar_for_stops(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(buy(10, stop=95))
    data.add_bar("AAA", MON, 100, 101, 94, 96)
    data.quotes["AAA"] = 96.0
    clock.t = utc(2026, 10, 19, 21)
    assert "AAA" not in broker.get_positions()
    fills = broker.get_fills()
    assert [f.reason for f in fills] == ["order", "stop"]
    assert fills[-1].price == pytest.approx(95 * (1 - SLIP))


def test_limit_day_orders_fill_on_a_cross_or_expire(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    o = broker.submit_order(buy(10, order_type=OrderType.LIMIT, limit_price=90))
    assert o.status == OrderStatus.PENDING
    clock.t = utc(2026, 10, 19, 21)
    assert broker.get_open_orders() == [] and not broker.get_positions()  # expired

    clock.t = utc(2026, 10, 20, 15)
    broker.submit_order(buy(10, order_type=OrderType.LIMIT, limit_price=90))
    data.quotes["AAA"] = 89.0
    assert broker.get_positions()["AAA"].avg_price == 89.0


def test_queued_limit_order_uses_the_sessions_range(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(buy(10, order_type=OrderType.LIMIT, limit_price=97))
    clock.t = utc(2026, 10, 19, 21)
    data.add_bar("AAA", MON, 99, 100, 96, 98)
    data.quotes["AAA"] = 98.0
    assert broker.get_positions()["AAA"].avg_price == 97


def test_day_pnl_is_measured_from_the_previous_close(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10))
    data.add_bar("AAA", MON, 99, 111, 98, 110)
    clock.t = utc(2026, 10, 20, 15)
    data.add_bar("AAA", TUE, 112, 121, 111, 120)
    data.quotes["AAA"] = 120.0
    a = broker.get_account()
    assert a.day_start_equity == pytest.approx(a.cash + 10 * 110)
    assert a.equity == pytest.approx(a.cash + 10 * 120)


def test_close_all_while_closed_queues_sells_and_reset_wipes(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10))
    clock.t = utc(2026, 10, 19, 22)
    broker.close_all_positions()
    pending = broker.get_open_orders()
    assert len(pending) == 1 and pending[0].request.side == Side.SELL
    broker.reset(50_000)
    assert broker.get_account().equity == 50_000 and broker.get_open_orders() == []


def test_router_and_risk_work_on_top_of_the_local_account(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    router = OrderRouter(broker, RiskManager(RiskLimits(), allowed_symbols=["AAA"]), Journal(":memory:"))
    rejected = router.route(buy(10))  # no stop
    ok = router.route(buy(10, stop=92))
    assert not rejected.decision.approved and ok.order.status == OrderStatus.FILLED
