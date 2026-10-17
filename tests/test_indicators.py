import math

from agentrade import indicators as ind
from conftest import make_bars


def test_sma_and_short_history():
    assert ind.sma([1, 2, 3, 4], 2) == 3.5
    assert ind.sma([1, 2], 3) is None


def test_ema_converges_to_constant():
    assert math.isclose(ind.ema([5.0] * 30, 10), 5.0)


def test_rsi_extremes():
    assert ind.rsi(list(range(1, 30)), 14) == 100.0
    assert ind.rsi(list(range(30, 1, -1)), 14) < 1
    assert ind.rsi([1, 2], 14) is None


def test_atr_constant_range():
    bars = make_bars([100.0] * 20, spread=0.01)  # high 101, low 99 every day
    assert math.isclose(ind.atr(bars, 14), 2.0, rel_tol=1e-9)


def test_pct_return_and_volatility():
    assert math.isclose(ind.pct_return([100, 110, 121], 2), 0.21)
    assert ind.annualized_volatility([100.0] * 30) == 0.0
