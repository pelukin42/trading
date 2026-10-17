"""Rule-based baseline strategies. The agent has to beat these to justify its cost."""

from __future__ import annotations

import math
from typing import Protocol

from . import indicators as ind
from .config import RiskLimits
from .models import Bar, OrderRequest, Side
from .router import OrderRouter, RouteResult


class Strategy(Protocol):
    name: str
    warmup: int

    def target_weights(self, bars: dict[str, list[Bar]]) -> dict[str, float]:
        """Desired portfolio weight per symbol, from data up to the current bar."""
        ...


class EqualWeight:
    """Always invested, equal weight in every symbol (re-entered after stop-outs)."""

    name = "equal_weight"
    warmup = 15  # enough bars for the ATR used to place stops

    def target_weights(self, bars):
        return {s: 1 / len(bars) for s in bars}


class SmaCross:
    """Hold a symbol while its fast SMA is above its slow SMA."""

    name = "sma_cross"

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast, self.slow = fast, slow
        self.warmup = slow

    def target_weights(self, bars):
        on = []
        for s, b in bars.items():
            closes = [x.close for x in b]
            f, sl = ind.sma(closes, self.fast), ind.sma(closes, self.slow)
            if f is not None and sl is not None and f > sl:
                on.append(s)
        return {s: (1 / len(bars) if s in on else 0.0) for s in bars}


class Momentum:
    """Equal-weight the top-k symbols by trailing return, only if that return is positive."""

    name = "momentum"

    def __init__(self, lookback: int = 120, top_k: int = 3):
        self.lookback, self.top_k = lookback, top_k
        self.warmup = lookback + 1

    def target_weights(self, bars):
        scores = {}
        for s, b in bars.items():
            r = ind.pct_return([x.close for x in b], self.lookback)
            if r is not None and r > 0:
                scores[s] = r
        top = sorted(scores, key=scores.get, reverse=True)[: self.top_k]
        return {s: (1 / self.top_k if s in top else 0.0) for s in bars}


STRATEGIES = {"equal_weight": EqualWeight, "sma_cross": SmaCross, "momentum": Momentum}


def rebalance(router: OrderRouter, bars: dict[str, list[Bar]], targets: dict[str, float],
              limits: RiskLimits, min_trade_pct: float = 0.02, stop_atr_mult: float = 3.0) -> list[RouteResult]:
    """Turn target weights into risk-checked orders. Weights are clipped to the risk limits first."""
    broker = router.broker
    account = broker.get_account()
    positions = broker.get_positions()
    equity = account.equity

    clipped = {s: min(max(w, 0.0), limits.max_position_pct * 0.98) for s, w in targets.items()}
    total = sum(clipped.values())
    cap = limits.max_gross_exposure_pct * 0.97
    if total > cap:
        clipped = {s: w * cap / total for s, w in clipped.items()}

    sells, buys = [], []
    for s, w in clipped.items():
        price = bars[s][-1].close
        held = positions[s].qty if s in positions else 0.0
        diff_value = w * equity - held * price
        if abs(diff_value) < min_trade_pct * equity and not (w == 0 and held > 0):
            continue
        qty = math.floor(abs(diff_value) / price)
        if diff_value < 0:
            qty = min(held, qty) if w > 0 else held
            if qty > 0:
                sells.append(OrderRequest(s, Side.SELL, qty, rationale=f"rebalance to {w:.1%}"))
        elif qty > 0:
            qty = min(qty, math.floor(limits.max_order_notional * 0.99 / price))  # the rest goes next bar
            if qty <= 0:
                continue
            stop = None
            if limits.require_stop_loss:
                a = ind.atr(bars[s], 14) or price * 0.05
                stop = round(max(price - stop_atr_mult * a, price * (1 - limits.max_stop_distance_pct * 0.95)), 2)
            buys.append(OrderRequest(s, Side.BUY, qty, stop_loss_price=stop, rationale=f"rebalance to {w:.1%}"))
    return [router.route(o) for o in sells + buys]
