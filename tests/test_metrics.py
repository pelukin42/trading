import math

import numpy as np

import pandas as pd
import pytest

from papertrade.metrics import ReportInput, build_report, max_drawdown, summarize


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_max_drawdown():
    assert max_drawdown(series([100, 120, 90, 130])) == pytest.approx(-0.25)


def test_summarize_basic_values():
    stats = summarize(series([100, 110]))
    assert stats["total_return"] == pytest.approx(0.10)
    assert stats["final_value"] == 110
    assert math.isnan(stats["sharpe"])  # un solo retorno: no hay desviación estándar


def test_report_says_no_when_benchmark_wins_and_deducts_llm_cost():
    equity = series([10_000, 10_500, 11_000])
    bench = series([100, 110, 120])
    llm = pd.Series([200.0], index=[equity.index[1]])
    text, summary = build_report(
        ReportInput("t", equity, bench, "SPY", n_trades=3, commissions=3.0, slippage=1.5, llm_costs=llm)
    )
    assert "Neta de IA" in text
    assert "Costo de IA (estimado): $200.00" in text
    assert summary["stats"]["Neta de IA"]["total_return"] == pytest.approx(0.08)
    assert summary["diff_vs_benchmark"] == pytest.approx(0.08 - 0.20)
    assert not summary["beats_benchmark"]
    assert not summary["conclusive"]
    assert "Veredicto: sin conclusión todavía" in text
    assert "muestra demasiado corta" in text


def ramp(start, end, n=200):
    return series(np.linspace(start, end, n))


def test_report_says_yes_when_strategy_wins_over_long_sample():
    text, summary = build_report(ReportInput("t", ramp(100, 130), ramp(50, 55), "SPY"))
    assert summary["beats_benchmark"] and summary["conclusive"]
    assert "Veredicto: SÍ agrega valor" in text
    assert "Neta de IA" not in text
    assert "AVISO" not in text


def test_report_compares_against_same_tickers_buy_and_hold():
    idx = pd.bdate_range("2024-01-01", periods=200)
    universe = pd.DataFrame({"A": np.linspace(10, 20, 200), "B": np.full(200, 10.0)}, index=idx)
    text, summary = build_report(
        ReportInput("t", ramp(100, 140), ramp(50, 55), "SPY", universe_close=universe)
    )
    assert summary["stats"]["Mismos tickers"]["total_return"] == pytest.approx(0.5)
    assert summary["diff_vs_universe"] == pytest.approx(0.4 - 0.5)
    assert "vs comprar y mantener los mismos tickers: -10.00 puntos porcentuales -> NO" in text
    assert "Veredicto: NO agrega valor" in text  # le gana a SPY pero no a sus propios tickers


def test_report_skips_same_tickers_when_universe_is_the_benchmark():
    universe = pd.DataFrame({"SPY": [50.0, 55.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    text, _ = build_report(ReportInput("t", series([100, 110]), series([50, 55]), "SPY", universe_close=universe))
    assert "Mismos tickers" not in text
