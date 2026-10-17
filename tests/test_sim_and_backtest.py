import pytest

from agentrade.backtest import Backtester, compute_metrics
from agentrade.broker.sim import SimBroker
from agentrade.config import RiskLimits
from agentrade.data import synthetic_history
from agentrade.models import OrderRequest, OrderStatus, OrderType, Side
from agentrade.strategies import STRATEGIES
from conftest import make_bars


def broker_for(closes, opens=None, **kw):
    return SimBroker({"AAA": make_bars(closes, opens=opens)}, initial_cash=10_000, commission_bps=0,
                     slippage_bps=0, **kw)


def test_market_order_fills_at_next_bar_open_not_current_close():
    b = broker_for([100, 110, 120], opens=[100, 105, 118])
    b.submit_order(OrderRequest("AAA", Side.BUY, 10))
    assert b.get_positions() == {}
    b.advance()
    pos = b.get_positions()["AAA"]
    assert pos.avg_price == 105 and pos.qty == 10
    assert b.cash == 10_000 - 1050


def test_bars_never_include_the_future():
    b = broker_for([100, 101, 102, 103])
    assert [x.close for x in b.get_bars("AAA")] == [100]
    b.advance()
    assert b.get_latest_price("AAA") == 101


def test_stop_loss_triggers_and_gaps_fill_at_open():
    b = broker_for([100, 100, 80], opens=[100, 100, 85])
    b.submit_order(OrderRequest("AAA", Side.BUY, 10, stop_loss_price=95))
    b.advance()  # filled at 100
    b.advance()  # opens at 85, below the 95 stop
    assert "AAA" not in b.get_positions()
    stop_fill = b.fills[-1]
    assert stop_fill.reason == "stop" and stop_fill.price == 85
    assert stop_fill.realized_pnl == pytest.approx(-150)


def test_sells_fund_buys_queued_in_the_same_step():
    h = {"AAA": make_bars([100, 100, 100]), "BBB": make_bars([50, 50, 50])}
    b = SimBroker(h, initial_cash=10_000, commission_bps=0, slippage_bps=0)
    b.submit_order(OrderRequest("AAA", Side.BUY, 100))
    b.advance()
    assert b.cash == 0
    b.submit_order(OrderRequest("BBB", Side.BUY, 200))
    b.submit_order(OrderRequest("AAA", Side.SELL, 100))
    b.advance()
    assert set(b.get_positions()) == {"BBB"}


def test_unfilled_limit_order_expires():
    b = broker_for([100, 100, 100])
    o = b.submit_order(OrderRequest("AAA", Side.BUY, 10, order_type=OrderType.LIMIT, limit_price=50))
    b.advance()
    assert o.status == OrderStatus.CANCELED and not b.get_open_orders()


def test_metrics_drawdown_and_return():
    curve = [(i, v) for i, v in enumerate([100, 120, 90, 110])]
    m = compute_metrics(curve, [])
    assert m.total_return == pytest.approx(0.10)
    assert m.max_drawdown == pytest.approx(-0.25)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_run_end_to_end_within_limits(name):
    history = synthetic_history(["AAA", "BBB", "CCC", "DDD"], days=400, seed=3)
    limits = RiskLimits()
    res = Backtester(history, limits).run_strategy(STRATEGIES[name]())
    assert res.decisions > 0
    assert len(res.equity_curve) > 100
    assert res.metrics.max_drawdown > -0.9
    # No order may have been submitted without a stop when stops are required.
    rows = res.journal.db.execute("SELECT stop_loss_price FROM orders WHERE side='buy' AND approved=1").fetchall()
    assert rows and all(r[0] is not None for r in rows)
