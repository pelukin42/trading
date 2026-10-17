import pandas as pd
import pytest

from papertrade.config import BrokerConfig, Config, RiskConfig
from papertrade.data import MarketData
from papertrade.engine import (
    Order,
    SimBroker,
    apply_risk,
    backtest_decider,
    decide_step,
    is_due,
    plan_orders,
    simulate,
)
from papertrade.strategies import Decision, Strategy


def make_cfg(**risk) -> Config:
    return Config(
        name="t",
        symbols=["A", "B"],
        strategy={"type": "buy_and_hold"},
        rebalance="daily",
        broker=BrokerConfig(starting_cash=1000, slippage_bps=100, commission_per_order=1.0),
        risk=RiskConfig(**risk),
    )


def test_apply_risk_caps_scales_and_drops_unknown():
    risk = RiskConfig(max_weight=0.6, cash_buffer=0.01)
    out = apply_risk({"A": 0.8, "B": 0.8, "X": 0.5, "C": -0.2}, risk, ["A", "B", "C"])
    assert out == pytest.approx({"A": 0.495, "B": 0.495, "C": 0.0})


def test_apply_risk_treats_nan_as_zero():
    out = apply_risk({"A": float("nan")}, RiskConfig(), ["A"])
    assert out == {"A": 0.0}


def test_plan_orders_sells_first_and_liquidates_untargeted():
    prices = {"A": 100.0, "Z": 50.0, "B": 20.0}
    orders = plan_orders({"A": 0.25, "B": 0.5}, {"A": 10, "Z": 5}, prices, equity=2000, band=0.02)
    assert orders == [Order("A", "sell", 5), Order("Z", "sell", 5, close_position=True), Order("B", "buy", 50)]


def test_plan_orders_skips_changes_inside_band():
    orders = plan_orders({"A": 0.51}, {"A": 10}, {"A": 100.0}, equity=2000, band=0.02)
    assert orders == []


def test_plan_orders_whole_shares_unless_fractional():
    assert plan_orders({"A": 0.5}, {}, {"A": 300.0}, equity=1000, band=0.02) == [Order("A", "buy", 1)]
    fractional = plan_orders({"A": 0.5}, {}, {"A": 300.0}, equity=1000, band=0.02, fractional=True)
    assert fractional[0].qty == pytest.approx(1.666666)


def test_sim_broker_applies_costs_and_never_overdraws():
    broker = SimBroker(BrokerConfig(starting_cash=1000, slippage_bps=100, commission_per_order=1.0))
    fill = broker.execute(Order("A", "buy", 20), 100.0, "2024-01-02")
    assert fill.qty == 9  # (1000 - 1) / 101 = 9.89 acciones
    assert fill.price == pytest.approx(101.0)
    assert fill.slippage_cost == pytest.approx(9.0)
    assert broker.cash == pytest.approx(1000 - 909 - 1)

    sell = broker.execute(Order("A", "sell", 50), 100.0, "2024-01-03")
    assert sell.qty == 9
    assert sell.price == pytest.approx(99.0)
    assert "A" not in broker.positions
    assert broker.cash == pytest.approx(90 + 891 - 1)


def test_is_due():
    mon, tue, next_mon = pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-04"), pd.Timestamp("2024-06-10")
    assert is_due(mon, None, "weekly")
    assert not is_due(tue, mon, "weekly")
    assert is_due(next_mon, mon, "weekly")
    assert is_due(tue, mon, "daily")
    assert not is_due(mon, mon, "daily")
    assert not is_due(pd.Timestamp("2024-06-28"), mon, "monthly")
    assert is_due(pd.Timestamp("2024-07-01"), mon, "monthly")


class Fixed(Strategy):
    def __init__(self, symbols, weights):
        super().__init__(symbols)
        self.weights = weights
        self.seen = []

    def decide(self, as_of, data, current):
        self.seen.append((as_of, data.close.index.max()))
        return Decision(dict(self.weights))


def test_kill_switch_liquidates_once_and_skips_strategy():
    cfg = make_cfg(max_drawdown_stop=0.25)
    strategy = Fixed(["A", "B"], {"A": 0.5})
    day = pd.Timestamp("2024-06-03")
    data = MarketData(pd.DataFrame({"A": [1.0]}, index=[day]), pd.DataFrame({"A": [1.0]}, index=[day]))

    decision = decide_step(strategy, cfg, day, data, {"A": 0.6}, drawdown=0.3, last_decision=None)
    assert decision.weights == {"A": 0.0}
    assert "kill_switch" in decision.notes
    assert decide_step(strategy, cfg, day, data, {}, drawdown=0.3, last_decision=None) is None
    assert strategy.seen == []


def test_simulate_fills_at_next_open_without_lookahead():
    days = pd.bdate_range("2024-01-01", periods=4)
    data = MarketData(
        open=pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0], "B": [1.0] * 4}, index=days),
        close=pd.DataFrame({"A": [11.0, 21.0, 31.0, 41.0], "B": [1.0] * 4}, index=days),
    )
    cfg = make_cfg(cash_buffer=0.0)
    cfg.broker = BrokerConfig(starting_cash=1000, slippage_bps=0, commission_per_order=0)
    strategy = Fixed(["A", "B"], {"A": 0.5})

    result = simulate(cfg, data, days[0], backtest_decider(strategy, cfg))

    assert all(as_of == latest for as_of, latest in strategy.seen)
    first = result.fills[0]
    assert first.date == days[1].strftime("%Y-%m-%d")
    assert first.price == 20.0  # apertura del día siguiente, no el cierre de la señal
    assert first.qty == 45  # 50% de 1000 al cierre de 11 = 45 acciones
    assert result.equity["equity"].iloc[0] == 1000
