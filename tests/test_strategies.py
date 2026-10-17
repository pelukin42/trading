from enum import Enum

import numpy as np
import pandas as pd
import pytest

from papertrade.config import ConfigError
from papertrade.data import MarketData
from papertrade.strategies import (
    BuyAndHold,
    SmaCrossover,
    TradingAgentsStrategy,
    build_strategy,
    normalize_rating,
)


def market(columns: dict[str, np.ndarray]) -> MarketData:
    n = len(next(iter(columns.values())))
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame(columns, index=idx)
    return MarketData(close.copy(), close)


def test_sma_long_in_uptrend_flat_in_downtrend():
    data = market({"UP": np.linspace(50, 150, 250), "DOWN": np.linspace(150, 50, 250)})
    decision = SmaCrossover(["UP", "DOWN"], fast=20, slow=100).decide(data.dates[-1], data, {})
    assert decision.weights == {"UP": 0.5, "DOWN": 0.0}


def test_sma_waits_for_enough_history():
    data = market({"UP": np.linspace(50, 150, 30)})
    decision = SmaCrossover(["UP"], fast=5, slow=50).decide(data.dates[-1], data, {})
    assert decision.weights == {"UP": 0.0}


def test_buy_and_hold_buys_once_then_holds():
    strategy = BuyAndHold(["A", "B"])
    assert strategy.decide(None, None, {}).weights == {"A": 0.5, "B": 0.5}
    assert strategy.decide(None, None, {"A": 0.7, "B": 0.29}).weights == {"A": 0.7, "B": 0.29}


class Rating(Enum):
    BUY = "Buy"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Buy", "BUY"), (" overweight ", "OVERWEIGHT"), (Rating.BUY, "BUY"), ("REVIEW", "REVIEW"), ("no sé", "REVIEW")],
)
def test_normalize_rating(raw, expected):
    assert normalize_rating(raw) == expected


class FakeGraph:
    def __init__(self, ratings):
        self.ratings = ratings
        self.calls = []

    def propagate(self, symbol, day):
        self.calls.append((symbol, day))
        return {}, self.ratings[symbol]


def test_tradingagents_maps_ratings_to_weights():
    graph = FakeGraph({"AAA": "Buy", "BBB": "Hold", "CCC": "Underweight", "DDD": "garbage"})
    strategy = TradingAgentsStrategy(
        ["AAA", "BBB", "CCC", "DDD"], cost_per_call_usd=0.4, graph_factory=lambda overrides: graph
    )
    decision = strategy.decide(pd.Timestamp("2026-10-16"), None, {"BBB": 0.2, "DDD": 0.1})

    assert decision.weights == pytest.approx({"AAA": 0.25, "BBB": 0.2, "CCC": 0.0625, "DDD": 0.1})
    assert decision.notes["ratings"]["DDD"] == "REVIEW"
    assert decision.llm_cost_usd == pytest.approx(1.6)
    assert graph.calls[0] == ("AAA", "2026-10-16")
    assert strategy.uses_llm


def test_build_strategy_errors():
    with pytest.raises(ConfigError, match="desconocida"):
        build_strategy({"type": "magic"}, ["A"])
    with pytest.raises(ConfigError, match="inválidos"):
        build_strategy({"type": "sma_crossover", "rapida": 3}, ["A"])
    with pytest.raises(ConfigError):
        build_strategy({"type": "sma_crossover", "fast": 50, "slow": 20}, ["A"])
