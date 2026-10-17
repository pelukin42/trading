"""Métricas de rendimiento y el reporte comparativo contra el benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252
MIN_DAYS_FOR_CONCLUSION = 126  # ~6 meses de días hábiles
UNIVERSE_LABEL = "Mismos tickers"


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize(equity: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = len(returns) / TRADING_DAYS
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    sharpe = float("nan")
    if std and math.isfinite(std) and std > 0:
        sharpe = float((returns.mean() - risk_free_rate / TRADING_DAYS) / std * math.sqrt(TRADING_DAYS))
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": std * math.sqrt(TRADING_DAYS),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "final_value": float(equity.iloc[-1]),
    }


@dataclass
class ReportInput:
    title: str
    equity: pd.Series
    benchmark_close: pd.Series
    benchmark_name: str
    n_trades: int = 0
    commissions: float = 0.0
    slippage: float = 0.0
    llm_costs: pd.Series | None = None  # costo de IA por fecha de decisión
    universe_close: pd.DataFrame | None = None  # cierres de los tickers de la estrategia
    risk_free_rate: float = 0.0
    warnings: tuple[str, ...] = ()


def _pct(x: float) -> str:
    return "n/d" if not math.isfinite(x) else f"{x:+.2%}"


def _num(x: float) -> str:
    return "n/d" if not math.isfinite(x) else f"{x:.2f}"


def _days(n: int) -> str:
    return "1 día hábil" if n == 1 else f"{n} días hábiles"


def _money(x: float) -> str:
    return f"${x:,.2f}"


def build_report(r: ReportInput) -> tuple[str, dict]:
    equity = r.equity.astype(float)
    bench_close = r.benchmark_close.reindex(equity.index).ffill().bfill().astype(float)
    benchmark = equity.iloc[0] * bench_close / bench_close.iloc[0]

    columns = {"Estrategia": equity}
    llm_total = 0.0
    if r.llm_costs is not None and r.llm_costs.sum() > 0:
        cumulative = r.llm_costs.groupby(level=0).sum().reindex(equity.index, fill_value=0.0).cumsum()
        llm_total = float(cumulative.iloc[-1])
        columns["Neta de IA"] = equity - cumulative
    universe = r.universe_close
    if universe is not None and len(universe.columns) and list(universe.columns) != [r.benchmark_name]:
        universe = universe.reindex(equity.index).ffill().bfill().astype(float)
        columns[UNIVERSE_LABEL] = equity.iloc[0] * (universe / universe.iloc[0]).mean(axis=1)
    columns[f"Benchmark ({r.benchmark_name})"] = benchmark

    stats = {name: summarize(series, r.risk_free_rate) for name, series in columns.items()}
    rows = [
        ("Retorno total", "total_return", _pct),
        ("Retorno anualizado", "cagr", _pct),
        ("Volatilidad anual", "volatility", lambda x: "n/d" if not math.isfinite(x) else f"{x:.2%}"),
        ("Sharpe", "sharpe", _num),
        ("Caída máxima", "max_drawdown", lambda x: "n/d" if not math.isfinite(x) else f"{x:.2%}"),
        ("Valor final", "final_value", _money),
    ]

    n_days = len(equity)
    width = max(20, *(len(n) + 2 for n in columns))
    lines = [
        r.title,
        f"Periodo: {equity.index[0]:%Y-%m-%d} -> {equity.index[-1]:%Y-%m-%d} ({_days(n_days)})",
        "",
        " " * 22 + "".join(name.rjust(width) for name in columns),
    ]
    for label, key, fmt in rows:
        lines.append(label.ljust(22) + "".join(fmt(stats[name][key]).rjust(width) for name in columns))

    lines += [
        "",
        f"Operaciones: {r.n_trades}   Comisiones: {_money(r.commissions)}   Slippage: {_money(r.slippage)}",
    ]
    if llm_total:
        lines.append(f"Costo de IA (estimado): {_money(llm_total)}")

    net_name = "Neta de IA" if "Neta de IA" in columns else "Estrategia"
    bench_name = f"Benchmark ({r.benchmark_name})"
    diff = stats[net_name]["total_return"] - stats[bench_name]["total_return"]
    beats = diff > 0
    lines += [
        "",
        "Después de costos:",
        f"  vs {bench_name}: {diff * 100:+.2f} puntos porcentuales -> {'SÍ' if beats else 'NO'} le gana",
    ]
    diff_universe = None
    beats_universe = True
    if UNIVERSE_LABEL in columns:
        diff_universe = stats[net_name]["total_return"] - stats[UNIVERSE_LABEL]["total_return"]
        beats_universe = diff_universe > 0
        lines += [
            f"  vs comprar y mantener los mismos tickers: {diff_universe * 100:+.2f} puntos porcentuales"
            f" -> {'SÍ' if beats_universe else 'NO'} le gana",
            "  (Si no le gana a los mismos tickers, la ventaja viene de qué acciones se eligieron,",
            "   no de cómo opera la estrategia.)",
        ]
    conclusive = n_days >= MIN_DAYS_FOR_CONCLUSION
    if not conclusive:
        lines.append("Veredicto: sin conclusión todavía; la muestra es demasiado corta (ver aviso).")
    elif beats and beats_universe:
        lines.append("Veredicto: SÍ agrega valor en este periodo.")
    else:
        lines.append("Veredicto: NO agrega valor. Comprar y mantener habría dado más.")

    warnings = list(r.warnings)
    if n_days < MIN_DAYS_FOR_CONCLUSION:
        warnings.append(
            f"Solo {_days(n_days)}: es una muestra demasiado corta para sacar conclusiones "
            "(mínimo recomendado ~6 meses, idealmente más de un año)."
        )
    for w in warnings:
        lines.append(f"AVISO: {w}")

    summary = {
        "stats": stats,
        "diff_vs_benchmark": diff,
        "beats_benchmark": beats,
        "beats_universe": beats_universe,
        "diff_vs_universe": diff_universe,
        "conclusive": conclusive,
        "days": n_days,
    }
    return "\n".join(lines), summary
