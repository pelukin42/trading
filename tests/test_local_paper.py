"""LocalPaperBroker against a fake data feed and a controllable clock."""

from datetime import date, datetime, timezone

import pytest

from agentrade.broker.base import BrokerError
from agentrade.broker.local_paper import LocalPaperBroker
from agentrade.config import RiskLimits
from agentrade.journal import Journal
from agentrade.models import Bar, CorporateAction, OrderRequest, OrderStatus, OrderType, Side
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
        self.quote_times: dict[str, datetime] = {}  # default: printed "now"
        self.actions: dict[str, list[CorporateAction]] = {}

    def add_bar(self, symbol, d: date, o, h, l, c):
        bars = self.bars.setdefault(symbol, [])
        t = datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc)
        bars[:] = [b for b in bars if b.t != t] + [Bar(t, o, h, l, c, 1e6)]  # re-adding a day replaces it
        bars.sort(key=lambda b: b.t)

    def get_bars(self, symbol, limit=200):
        return [b for b in self.bars.get(symbol, []) if b.t <= self.clock()][-limit:]

    def get_quote(self, symbol):
        return Quote(self.quotes[symbol], self.quote_times.get(symbol, self.clock()))

    def get_corporate_actions(self, symbol, since):
        today = self.clock().date()
        return [a for a in self.actions.get(symbol, []) if since < a.session <= today]


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


# --- regressions from the adversarial review -----------------------------------------------

FRI = date(2026, 10, 16)


def hold_from_friday(clock, data, broker, qty=10, stop=95.0):
    """Enter intraday on Friday (its bar low 98 stays above the stop)."""
    clock.t = utc(2026, 10, 16, 15)
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(qty, stop=stop))


def test_intraday_add_first_checks_the_days_range_against_the_existing_stop(env):
    clock, data, broker, _ = env
    hold_from_friday(clock, data, broker)
    clock.t = utc(2026, 10, 19, 15)
    data.add_bar("AAA", MON, 99, 100, 94, 99)  # dipped to 94 this morning, back at 99 now
    data.quotes["AAA"] = 99.0
    o = broker.submit_order(buy(5, stop=96))
    assert o.status == OrderStatus.REJECTED
    assert "AAA" not in broker.get_positions()
    fills = broker.get_fills()
    assert [f.reason for f in fills] == ["order", "stop"] and fills[-1].price == pytest.approx(95 * (1 - SLIP))
    assert broker.recent_events()[-1]["status"] == "rejected"


def test_catch_up_reaches_back_further_than_ten_sessions(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(buy(10, stop=95))  # queued for Monday's open
    d = MON
    while d <= date(2026, 11, 6):
        if d.weekday() < 5:
            low = 90 if d == TUE else 99
            data.add_bar("AAA", d, 100, 101, low, 100)
        d = date.fromordinal(d.toordinal() + 1)
    clock.t = utc(2026, 11, 9, 15)
    data.quotes["AAA"] = 100.0
    assert broker.get_open_orders() == [] and broker.get_positions() == {}
    fills = broker.get_fills()
    assert [(f.reason, f.t.date()) for f in fills] == [("order", MON), ("stop", TUE)]


def test_missed_sessions_are_settled_before_the_new_day_pnl_baseline(env):
    clock, data, broker, _ = env
    hold_from_friday(clock, data, broker, qty=200, stop=88)
    data.add_bar("AAA", MON, 100, 101, 85, 100)  # stop hit Monday while nobody was looking
    data.add_bar("AAA", TUE, 100, 101, 99, 100)
    clock.t = utc(2026, 10, 20, 14, 30)
    data.quotes["AAA"] = 100.0
    a = broker.get_account()
    assert broker.get_positions() == {}
    assert a.day_pnl == pytest.approx(0.0)  # Monday's loss belongs to Monday


def test_market_order_before_the_first_print_takes_the_opening_price(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 19, 13, 30, 20)  # 9:30:20 ET
    data.quotes["AAA"] = 100.0               # still Friday's close
    data.quote_times["AAA"] = utc(2026, 10, 16, 20)
    o = broker.submit_order(buy(10))
    assert o.status == OrderStatus.PENDING
    data.add_bar("AAA", MON, 105, 106, 104, 105.5)
    data.quotes["AAA"] = 105.5
    data.quote_times.pop("AAA")
    clock.t = utc(2026, 10, 19, 13, 35)
    assert broker.get_positions()["AAA"].avg_price == pytest.approx(105 * (1 + SLIP))


def test_queued_orders_execute_in_submission_order_for_the_same_symbol(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)          # Saturday: buy...
    broker.submit_order(buy(10))
    clock.t = utc(2026, 10, 18, 15)          # ...Sunday: changed my mind
    broker.submit_order(OrderRequest("AAA", Side.SELL, 10))
    clock.t = utc(2026, 10, 19, 14)
    data.add_bar("AAA", MON, 100, 101, 99, 100)
    data.quotes["AAA"] = 100.0
    assert broker.get_positions() == {}
    assert [f.side for f in broker.get_fills()] == [Side.BUY, Side.SELL]


def test_a_stop_out_cancels_the_symbols_other_orders(env):
    clock, data, broker, _ = env
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10, stop=95))
    broker.submit_order(OrderRequest("AAA", Side.SELL, 10, order_type=OrderType.LIMIT, limit_price=110))
    data.quotes["AAA"] = 94.0
    assert broker.get_open_orders() == [] and broker.get_positions() == {}
    assert broker.recent_events()[-1]["status"] == "canceled"


def test_the_days_bar_is_only_judged_after_it_settles(env):
    clock, data, broker, _ = env
    hold_from_friday(clock, data, broker)
    data.add_bar("AAA", MON, 99, 100, 95.3, 99)  # provisional bar right after the close
    data.quotes["AAA"] = 99.0
    clock.t = utc(2026, 10, 19, 20, 5)
    assert "AAA" in broker.get_positions()
    data.add_bar("AAA", MON, 99, 100, 94.6, 99)  # final bar: the low went through the stop
    clock.t = utc(2026, 10, 19, 20, 30)
    assert "AAA" not in broker.get_positions()


def test_queued_market_order_expires_when_its_session_has_no_data(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(buy(10))
    clock.t = utc(2026, 10, 19, 21)  # Monday's session is over and AAA has no bar for it
    data.quotes["AAA"] = 100.0
    assert broker.get_open_orders() == []
    assert broker.recent_events()[-1]["status"] == "expired"


class LaggyData(FakeData):
    """Hides Monday's bar until the `reveal_at`-th get_bars call, to land a fill at any point of a read."""

    calls, reveal_at = 0, 10**9

    def get_bars(self, symbol, limit=200):
        self.calls += 1
        bars = super().get_bars(symbol, limit)
        return bars if self.calls >= self.reveal_at else [b for b in bars if b.t.date() != MON]


@pytest.mark.parametrize("reveal_at", [1, 2, 3, 4, 10**9])
@pytest.mark.parametrize("queued_side", ["buy", "sell"])
def test_a_fill_landing_while_the_router_reads_cannot_bypass_position_limits(env, reveal_at, queued_side):
    """Whenever a queued order fills relative to the router's reads, an oversized buy is rejected."""
    clock, data, broker, _ = env
    laggy = LaggyData(clock)
    laggy.bars, laggy.quotes = data.bars, {"AAA": 100.0}
    broker.data = laggy
    router = OrderRouter(broker, RiskManager(RiskLimits(), allowed_symbols=["AAA"]), Journal(":memory:"))
    clock.t = utc(2026, 10, 16, 15)  # Friday, open
    if queued_side == "sell":
        assert router.route(buy(200, stop=90)).order.status == OrderStatus.FILLED  # 20% of equity
    clock.t = utc(2026, 10, 16, 22)  # Friday evening: queue for Monday's open
    queued = buy(200, stop=90) if queued_side == "buy" else OrderRequest("AAA", Side.SELL, 100)
    assert router.route(queued).order.status == OrderStatus.PENDING

    clock.t = utc(2026, 10, 19, 14, 1)
    laggy.add_bar("AAA", MON, 100, 101, 99, 100)
    laggy.calls, laggy.reveal_at = 0, reveal_at
    held_after = 200 if queued_side == "buy" else 100
    oversized = buy(int(0.2 * 100_000 / 100) - held_after + 90, stop=90)  # 9 points over the 20% cap
    assert not router.route(oversized).decision.approved

    laggy.reveal_at = 0
    assert broker.get_positions()["AAA"].qty == held_after


def test_broker_side_rejections_are_journaled_as_not_approved_with_a_reason(env):
    clock, data, broker, _ = env
    journal = Journal(":memory:")
    router = OrderRouter(broker, RiskManager(RiskLimits(), allowed_symbols=["AAA"]), journal)
    clock.t = utc(2026, 10, 16, 15)
    data.quotes["AAA"] = 100.0
    router.route(buy(10, stop=95))
    clock.t = utc(2026, 10, 19, 15)
    data.add_bar("AAA", MON, 99, 100, 94, 99)  # this morning's dip already hit the stop
    data.quotes["AAA"] = 99.0
    result = router.route(buy(5, stop=96))
    assert result.to_dict()["status"] == "rejected" and "stopped out" in result.to_dict()["broker_reason"]
    assert journal.approved_orders_since(utc(2026, 10, 19)) == 0


def test_splits_are_applied_before_the_restated_history_is_read(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 16, 15)
    data.quotes["AAA"] = 1000.0
    broker.submit_order(buy(10, stop=900))
    # 10-for-1 split effective Monday: the feed restates every past bar in post-split shares.
    data.bars["AAA"] = [Bar(b.t, b.open / 10, b.high / 10, b.low / 10, b.close / 10, b.volume) for b in data.bars["AAA"]]
    data.add_bar("AAA", MON, 100, 101, 99, 100.5)
    data.actions["AAA"] = [CorporateAction(MON, "split", ratio=10.0)]
    clock.t = utc(2026, 10, 19, 21)
    data.quotes["AAA"] = 100.5
    pos = broker.get_positions()["AAA"]
    assert pos.qty == 100 and pos.avg_price == pytest.approx(100.05)
    assert broker.get_account().equity == pytest.approx(100_000 - 10 * 1000 * SLIP - 10 * 1000.5 * 0.0001 + 10 * 5,
                                                        abs=0.01)
    assert [f.reason for f in broker.get_fills()] == ["order"]  # no bogus stop-out


def test_dividends_are_credited_to_shares_held_at_the_previous_close(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 16, 15)
    data.quotes["AAA"] = 100.0
    broker.submit_order(buy(10, stop=90))
    cash_before = broker.get_account().cash
    data.add_bar("AAA", MON, 99.5, 100, 99, 99.6)
    data.actions["AAA"] = [CorporateAction(MON, "dividend", amount=0.5)]
    clock.t = utc(2026, 10, 19, 21)
    data.quotes["AAA"] = 99.6
    assert broker.get_account().cash == pytest.approx(cash_before + 5.0)
    clock.t = utc(2026, 10, 20, 21)  # credited once only
    assert broker.get_account().cash == pytest.approx(cash_before + 5.0)


def test_one_failed_fetch_does_not_skip_a_sessions_stop_check(env):
    clock, data, broker, _ = env
    hold_from_friday(clock, data, broker)
    data.add_bar("AAA", MON, 99, 100, 90, 99)  # breach on Monday
    data.add_bar("AAA", TUE, 99, 100, 98, 99)
    fail = {"left": 1}
    real_get_bars = data.get_bars

    def flaky_get_bars(symbol, limit=200):
        if fail["left"]:
            fail["left"] -= 1
            raise BrokerError("timeout")
        return real_get_bars(symbol, limit)

    data.get_bars = flaky_get_bars
    clock.t = utc(2026, 10, 21, 12)
    data.quotes["AAA"] = 99.0
    broker.get_account()  # this look hit the timeout
    assert "AAA" not in broker.get_positions()  # the next look still catches Monday's breach


def test_orders_on_a_symbol_without_data_expire_after_a_grace_period(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 17, 15)
    broker.submit_order(OrderRequest("GONE", Side.BUY, 10, stop_loss_price=1.0))
    clock.t = utc(2026, 10, 20, 15)  # Monday's session plus a day has passed, and GONE has no data at all
    data.quotes["AAA"] = 100.0
    assert broker.get_open_orders() == []
    assert broker.recent_events()[-1]["status"] == "expired"


def test_reset_recovers_a_damaged_state_file(env):
    clock, data, broker, tmp_path = env
    (tmp_path / "paper.json").write_text('{"version": 1, "cash": 5')
    with pytest.raises(BrokerError, match="reset-paper"):
        broker.get_account()
    fresh = LocalPaperBroker(data, tmp_path / "paper.json", initial_cash=10_000, now=clock, fresh=True)
    assert fresh.get_account().equity == 10_000


def test_a_split_is_inferred_when_the_feed_restates_history_without_an_event(env):
    clock, data, broker, _ = env
    clock.t = utc(2026, 10, 16, 15)
    data.quotes["AAA"] = 1000.0
    data.bars["AAA"] = [Bar(b.t, 990, 1010, 980, 1000, b.volume) for b in data.bars["AAA"]]
    data.add_bar("AAA", date(2026, 10, 15), 990, 1010, 980, 1000)
    broker.submit_order(buy(10, stop=900))
    # 15-for-1 split effective Monday; history restated, but no split event published (yet).
    data.bars["AAA"] = [Bar(b.t, b.open / 15, b.high / 15, b.low / 15, b.close / 15, b.volume)
                        for b in data.bars["AAA"]]
    data.add_bar("AAA", MON, 66.5, 67.5, 66, 67)
    clock.t = utc(2026, 10, 19, 21)
    data.quotes["AAA"] = 67.0
    pos = broker.get_positions()["AAA"]
    assert pos.qty == pytest.approx(150) and pos.avg_price == pytest.approx(1000 * (1 + SLIP) / 15)
    assert [f.reason for f in broker.get_fills()] == ["order"]
    assert broker.recent_events()[-1]["status"] == "split"
