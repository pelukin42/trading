"""Carga y valida la configuración TOML de cada experimento."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

REBALANCE_OPTIONS = ("daily", "weekly", "monthly")
BROKER_OPTIONS = ("sim", "alpaca")
DATA_OPTIONS = ("yfinance", "alpaca", "csv")


class ConfigError(ValueError):
    pass


@dataclass
class BrokerConfig:
    type: str = "sim"
    starting_cash: float = 10_000.0
    commission_per_order: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    fractional: bool = False


@dataclass
class DataConfig:
    source: str = "yfinance"
    lookback_days: int = 400
    csv_dir: str = "data"
    # Las cuentas "paper only" de Alpaca solo tienen derecho a datos IEX.
    feed: str = "iex"


@dataclass
class RiskConfig:
    max_weight: float = 1.0
    cash_buffer: float = 0.005
    rebalance_band: float = 0.02
    # 0 desactiva el kill switch; 0.25 liquida todo si la caída desde el máximo llega a 25%.
    max_drawdown_stop: float = 0.0


@dataclass
class Config:
    name: str
    symbols: list[str]
    strategy: dict
    benchmark: str = "SPY"
    rebalance: str = "weekly"
    risk_free_rate: float = 0.0
    runs_dir: str = "runs"
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.name

    @property
    def data_symbols(self) -> list[str]:
        return list(dict.fromkeys([*self.symbols, self.benchmark]))


def _build(cls, raw: dict, section: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Claves desconocidas en [{section}]: {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"Falta una clave obligatoria en [{section}]: {exc}") from None


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"No existe el archivo de configuración: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML inválido en {path}: {exc}") from None

    sections = {
        "broker": (BrokerConfig, raw.pop("broker", {})),
        "data": (DataConfig, raw.pop("data", {})),
        "risk": (RiskConfig, raw.pop("risk", {})),
    }
    built = {name: _build(cls, values, name) for name, (cls, values) in sections.items()}
    cfg = _build(Config, {**raw, **built}, "raíz")
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.name or "/" in cfg.name or "\\" in cfg.name:
        raise ConfigError("`name` es obligatorio y no puede contener barras")
    if not isinstance(cfg.symbols, list) or not cfg.symbols or not all(isinstance(s, str) for s in cfg.symbols):
        raise ConfigError('`symbols` debe ser una lista de tickers, p. ej. ["AAPL", "MSFT"]')
    cfg.symbols = [s.strip().upper() for s in cfg.symbols]
    cfg.benchmark = cfg.benchmark.strip().upper()
    if not isinstance(cfg.strategy, dict) or "type" not in cfg.strategy:
        raise ConfigError("Falta `type` en [strategy]")
    if cfg.rebalance not in REBALANCE_OPTIONS:
        raise ConfigError(f"`rebalance` debe ser uno de {REBALANCE_OPTIONS}")
    if cfg.broker.type not in BROKER_OPTIONS:
        raise ConfigError(f"[broker] type debe ser uno de {BROKER_OPTIONS}")
    if cfg.data.source not in DATA_OPTIONS:
        raise ConfigError(f"[data] source debe ser uno de {DATA_OPTIONS}")
    if not 0 < cfg.risk.max_weight <= 1:
        raise ConfigError("[risk] max_weight debe estar entre 0 y 1")
    if not 0 <= cfg.risk.cash_buffer < 1:
        raise ConfigError("[risk] cash_buffer debe estar entre 0 y 1")
    if not 0 <= cfg.risk.max_drawdown_stop < 1:
        raise ConfigError("[risk] max_drawdown_stop debe estar entre 0 y 1 (0 = desactivado)")
    if cfg.broker.starting_cash <= 0:
        raise ConfigError("[broker] starting_cash debe ser positivo")


def load_env_file(path: str | Path = ".env") -> None:
    """Carga variables KEY=VALUE de un .env sin pisar las que ya existen."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))
