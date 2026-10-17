"""Estrategias: cada una decide los pesos objetivo del portafolio al cierre de un día.

Para conectar tu propio agente, hereda de `Strategy`, implementa `decide` y
regístralo en `STRATEGIES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from papertrade.config import ConfigError
from papertrade.data import MarketData


@dataclass
class Decision:
    """Pesos objetivo por ticker (0.25 = 25% del capital). Lo que no se asigna queda en efectivo."""

    weights: dict[str, float]
    notes: dict = field(default_factory=dict)
    llm_cost_usd: float = 0.0


class Strategy:
    # True si la estrategia usa un LLM: sus backtests están contaminados (ver README).
    uses_llm = False

    def __init__(self, symbols: list[str]):
        self.symbols = list(symbols)

    def decide(self, as_of: pd.Timestamp, data: MarketData, current: dict[str, float]) -> Decision:
        """`data` solo contiene precios hasta `as_of`; `current` son los pesos actuales."""
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Compra partes iguales el primer día y no vuelve a operar."""

    def decide(self, as_of, data, current):
        if any(w > 0 for w in current.values()):
            return Decision(dict(current), {"accion": "mantener"})
        slot = 1 / len(self.symbols)
        return Decision({s: slot for s in self.symbols}, {"accion": "compra inicial"})


class SmaCrossover(Strategy):
    """Invierte en cada ticker solo si su media móvil rápida está sobre la lenta."""

    def __init__(self, symbols, fast: int = 50, slow: int = 200):
        super().__init__(symbols)
        if not 0 < fast < slow:
            raise ConfigError("sma_crossover requiere 0 < fast < slow")
        self.fast, self.slow = fast, slow

    def decide(self, as_of, data, current):
        closes = data.close[self.symbols]
        if len(closes) < self.slow:
            return Decision({s: 0.0 for s in self.symbols}, {"motivo": "historial insuficiente"})
        fast = closes.iloc[-self.fast :].mean()
        slow = closes.iloc[-self.slow :].mean()
        slot = 1 / len(self.symbols)
        weights = {s: slot if fast[s] > slow[s] else 0.0 for s in self.symbols}
        signals = {s: "alcista" if weights[s] else "bajista" for s in self.symbols}
        return Decision(weights, {"senales": signals})


RATING_FACTORS = {"BUY": 1.0, "OVERWEIGHT": 0.75, "UNDERWEIGHT": 0.25, "SELL": 0.0}
KEEP_RATINGS = {"HOLD", "REVIEW"}


def normalize_rating(raw) -> str:
    """Convierte la salida de TradingAgents (texto o enum) a BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL/REVIEW."""
    text = str(getattr(raw, "value", raw)).strip().upper().split(".")[-1]
    return text if text in RATING_FACTORS or text in KEEP_RATINGS else "REVIEW"


class TradingAgentsStrategy(Strategy):
    """Adaptador para TauricResearch/TradingAgents (https://github.com/TauricResearch/TradingAgents).

    Cada ticker recibe hasta 1/N del capital según el rating:
    Buy = 100% del cupo, Overweight = 75%, Underweight = 25%, Sell = 0%.
    Hold o un rating ilegible (REVIEW) mantienen el peso actual.
    """

    uses_llm = True

    def __init__(
        self,
        symbols,
        cost_per_call_usd: float = 0.0,
        debug: bool = False,
        config: dict | None = None,
        graph_factory: Callable[[dict], object] | None = None,
    ):
        super().__init__(symbols)
        self.cost_per_call_usd = cost_per_call_usd
        self._debug = debug
        self._overrides = dict(config or {})
        self._factory = graph_factory
        self._graph = None

    def _get_graph(self):
        if self._graph is not None:
            return self._graph
        if self._factory is not None:
            self._graph = self._factory(self._overrides)
            return self._graph
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
            from tradingagents.graph.trading_graph import TradingAgentsGraph
        except ImportError:
            raise ConfigError(
                "TradingAgents no está instalado. Instálalo con:\n"
                "  git clone https://github.com/TauricResearch/TradingAgents.git\n"
                "  pip install ./TradingAgents"
            ) from None
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(self._overrides)
        self._graph = TradingAgentsGraph(debug=self._debug, config=cfg)
        return self._graph

    def decide(self, as_of, data, current):
        graph = self._get_graph()
        slot = 1 / len(self.symbols)
        weights, ratings = {}, {}
        for symbol in self.symbols:
            _, raw = graph.propagate(symbol, as_of.strftime("%Y-%m-%d"))
            rating = normalize_rating(raw)
            ratings[symbol] = rating
            if rating in RATING_FACTORS:
                weights[symbol] = RATING_FACTORS[rating] * slot
            else:
                weights[symbol] = current.get(symbol, 0.0)
        cost = self.cost_per_call_usd * len(self.symbols)
        return Decision(weights, {"ratings": ratings}, llm_cost_usd=cost)


STRATEGIES: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHold,
    "sma_crossover": SmaCrossover,
    "tradingagents": TradingAgentsStrategy,
}


def build_strategy(spec: dict, symbols: list[str]) -> Strategy:
    params = dict(spec)
    kind = params.pop("type")
    cls = STRATEGIES.get(kind)
    if cls is None:
        raise ConfigError(f"Estrategia desconocida: {kind}. Opciones: {', '.join(STRATEGIES)}")
    try:
        return cls(symbols, **params)
    except TypeError as exc:
        raise ConfigError(f"Parámetros inválidos para la estrategia {kind}: {exc}") from None
