"""Backtester: replays history through SimBroker with the same router and risk engine used live."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from .broker.sim import SimBroker
from .config import RiskLimits
from .indicators import TRADING_DAYS
from .journal import Journal
from .models import Bar, Fill, Side
from .risk import RiskManager
from .router import OrderRouter
from .strategies import Strategy, rebalance

# Called once per decision step with the simulated broker and the risk-checked router.
Decider = Callable[[SimBroker, OrderRouter], None]


@dataclass
class Metrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float | None
    stop_outs: int

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Retorno total", f"{self.total_return:+.2%}"),
            ("CAGR", f"{self.cagr:+.2%}"),
            ("Volatilidad anual", f"{self.volatility:.2%}"),
            ("Sharpe (rf=0)", f"{self.sharpe:.2f}"),
            ("Máx. drawdown", f"{self.max_drawdown:.2%}"),
            ("Operaciones", str(self.trades)),
            ("Tasa de acierto", "n/a" if self.win_rate is None else f"{self.win_rate:.1%}"),
            ("Salidas por stop", str(self.stop_outs)),
        ]


def compute_metrics(curve: list[tuple[datetime, float]], fills: list[Fill],
                    periods_per_year: int = TRADING_DAYS) -> Metrics:
    values = [v for _, v in curve]
    rets = [b / a - 1 for a, b in zip(values[:-1], values[1:]) if a > 0]
    total = values[-1] / values[0] - 1 if values and values[0] else 0.0
    years = max(len(rets) / periods_per_year, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        std = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
    else:
        mean, std = 0.0, 0.0
    vol = std * math.sqrt(periods_per_year)
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    peak, mdd = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    exits = [f for f in fills if f.side == Side.SELL]
    win_rate = sum(f.realized_pnl > 0 for f in exits) / len(exits) if exits else None
    return Metrics(total, cagr, vol, sharpe, mdd, len(fills), win_rate, sum(f.reason == "stop" for f in exits))


def buy_and_hold_curve(history: dict[str, list[Bar]], start: datetime,
                       initial_cash: float) -> list[tuple[datetime, float]]:
    """Equal-weight buy & hold benchmark, no costs, no stops."""
    series = {s: {b.t: b.close for b in bars if b.t >= start} for s, bars in history.items()}
    times = sorted({t for s in series.values() for t in s})
    alloc = initial_cash / len(series)
    shares = {s: alloc / next(iter(v.values())) for s, v in series.items() if v}
    last = {s: next(iter(v.values())) for s, v in series.items() if v}
    curve = []
    for t in times:
        for s in shares:
            last[s] = series[s].get(t, last[s])
        curve.append((t, sum(shares[s] * last[s] for s in shares)))
    return curve


@dataclass
class BacktestResult:
    metrics: Metrics
    benchmark: Metrics
    equity_curve: list[tuple[datetime, float]]
    fills: list[Fill]
    rejected_orders: int
    decisions: int
    journal: Journal = field(repr=False)


class Backtester:
    def __init__(self, history: dict[str, list[Bar]], limits: RiskLimits, initial_cash: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 5.0, journal: Journal | None = None):
        self.history = history
        self.limits = limits
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.journal = journal or Journal(":memory:")

    def make_stack(self, warmup: int) -> tuple[SimBroker, OrderRouter]:
        broker = SimBroker(self.history, self.initial_cash, self.commission_bps, self.slippage_bps,
                           start_index=warmup)
        risk = RiskManager(self.limits, allowed_symbols=list(self.history))
        return broker, OrderRouter(broker, risk, self.journal)

    def run(self, decide: Decider, warmup: int, every: int = 1, max_decisions: int | None = None,
            broker: SimBroker | None = None, router: OrderRouter | None = None) -> BacktestResult:
        if broker is None or router is None:
            broker, router = self.make_stack(warmup)
        if warmup >= len(broker.timeline) - 1:
            raise ValueError(f"Not enough history: need more than {warmup} bars")
        start = broker.now
        decisions, step = 0, 0
        while True:
            if step % every == 0 and (max_decisions is None or decisions < max_decisions):
                decide(broker, router)
                decisions += 1
            step += 1
            if not broker.advance():
                break
        rejected = self.journal.db.execute("SELECT COUNT(*) FROM orders WHERE approved=0").fetchone()[0]
        bench = buy_and_hold_curve(self.history, start, self.initial_cash)
        return BacktestResult(
            metrics=compute_metrics(broker.equity_curve, broker.fills),
            benchmark=compute_metrics(bench, []),
            equity_curve=broker.equity_curve, fills=broker.fills,
            rejected_orders=int(rejected), decisions=decisions, journal=self.journal,
        )

    def run_strategy(self, strategy: Strategy, every: int = 1) -> BacktestResult:
        def decide(broker: SimBroker, router: OrderRouter) -> None:
            bars = {s: broker.get_bars(s, limit=strategy.warmup + 1) for s in self.history}
            bars = {s: b for s, b in bars.items() if b}
            rebalance(router, bars, strategy.target_weights(bars), self.limits)

        return self.run(decide, warmup=strategy.warmup, every=every)
