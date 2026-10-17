"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

LIVE_CONFIRMATION = "I_UNDERSTAND_REAL_MONEY_IS_AT_RISK"


class ConfigError(ValueError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader. Never overrides variables already set in the environment."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [s.strip().upper() for s in _env(name, default).split(",") if s.strip()]


@dataclass
class RiskLimits:
    """Hard limits enforced in code on every order, regardless of what the agent wants."""

    max_position_pct: float = 0.20        # max value of one symbol as a fraction of equity
    max_gross_exposure_pct: float = 1.00  # sum of long positions / equity (1.0 = no leverage)
    max_open_positions: int = 8
    max_order_notional: float = 25_000.0  # USD per order
    max_daily_loss_pct: float = 0.03      # stop opening risk after -3% on the day
    max_orders_per_day: int = 20
    require_stop_loss: bool = True        # every new long must carry a protective stop
    max_stop_distance_pct: float = 0.15   # stop can't be further than 15% below entry
    allow_short: bool = False

    @classmethod
    def from_env(cls) -> "RiskLimits":
        d = cls()
        return cls(
            max_position_pct=_float("AGENTRADE_MAX_POSITION_PCT", d.max_position_pct),
            max_gross_exposure_pct=_float("AGENTRADE_MAX_GROSS_EXPOSURE_PCT", d.max_gross_exposure_pct),
            max_open_positions=_int("AGENTRADE_MAX_OPEN_POSITIONS", d.max_open_positions),
            max_order_notional=_float("AGENTRADE_MAX_ORDER_NOTIONAL", d.max_order_notional),
            max_daily_loss_pct=_float("AGENTRADE_MAX_DAILY_LOSS_PCT", d.max_daily_loss_pct),
            max_orders_per_day=_int("AGENTRADE_MAX_ORDERS_PER_DAY", d.max_orders_per_day),
            require_stop_loss=_bool("AGENTRADE_REQUIRE_STOP_LOSS", d.require_stop_loss),
            max_stop_distance_pct=_float("AGENTRADE_MAX_STOP_DISTANCE_PCT", d.max_stop_distance_pct),
            allow_short=_bool("AGENTRADE_ALLOW_SHORT", d.allow_short),
        )


@dataclass
class Settings:
    mode: str = "paper"  # "paper" | "live"
    # "local": built-in paper account priced with Yahoo Finance data (no broker sign-up needed).
    # "alpaca": Alpaca paper or live account.
    broker: str = "local"
    paper_initial_cash: float = 100_000.0
    watchlist: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN"])
    data_dir: Path = Path("data")

    model: str = "claude-opus-5-5"
    effort: str = "medium"           # low | medium | high | xhigh | max
    max_agent_steps: int = 25        # tool-use round trips per cycle
    enable_web_search: bool = False  # let the agent read news via Anthropic's server-side web search
    report_language: str = "español"

    cycle_minutes: int = 60
    max_cycles_per_day: int = 8

    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""
    alpaca_data_feed: str = "iex"  # "iex" is free; "sip" needs a paid Alpaca data plan

    risk: RiskLimits = field(default_factory=RiskLimits)

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            mode=_env("AGENTRADE_MODE", d.mode).lower(),
            broker=_env("AGENTRADE_BROKER", d.broker).lower(),
            paper_initial_cash=_float("AGENTRADE_PAPER_CASH", d.paper_initial_cash),
            watchlist=_list("AGENTRADE_WATCHLIST", ",".join(d.watchlist)),
            data_dir=Path(_env("AGENTRADE_DATA_DIR", str(d.data_dir))),
            model=_env("AGENTRADE_MODEL", d.model),
            effort=_env("AGENTRADE_EFFORT", d.effort),
            max_agent_steps=_int("AGENTRADE_MAX_AGENT_STEPS", d.max_agent_steps),
            enable_web_search=_bool("AGENTRADE_ENABLE_WEB_SEARCH", d.enable_web_search),
            report_language=_env("AGENTRADE_REPORT_LANGUAGE", d.report_language),
            cycle_minutes=_int("AGENTRADE_CYCLE_MINUTES", d.cycle_minutes),
            max_cycles_per_day=_int("AGENTRADE_MAX_CYCLES_PER_DAY", d.max_cycles_per_day),
            alpaca_key_id=_env("ALPACA_API_KEY_ID", ""),
            alpaca_secret_key=_env("ALPACA_API_SECRET_KEY", ""),
            alpaca_data_feed=_env("ALPACA_DATA_FEED", d.alpaca_data_feed),
            risk=RiskLimits.from_env(),
        )

    @property
    def kill_switch_path(self) -> Path:
        return self.data_dir / "KILL"

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "journal.sqlite3"

    @property
    def paper_state_path(self) -> Path:
        return self.data_dir / "paper_account.json"

    def validate_live(self) -> None:
        """Live trading requires an explicit, typed confirmation in the environment."""
        if self.mode not in {"paper", "live"}:
            raise ConfigError(f"AGENTRADE_MODE must be 'paper' or 'live', got {self.mode!r}")
        if self.broker not in {"local", "alpaca"}:
            raise ConfigError(f"AGENTRADE_BROKER must be 'local' or 'alpaca', got {self.broker!r}")
        if self.mode == "live" and self.broker != "alpaca":
            raise ConfigError("Live trading needs a real broker: set AGENTRADE_BROKER=alpaca.")
        if self.mode == "live" and os.environ.get("AGENTRADE_LIVE_CONFIRM") != LIVE_CONFIRMATION:
            raise PermissionError(
                "Live mode is locked. Set AGENTRADE_LIVE_CONFIRM="
                f"{LIVE_CONFIRMATION} once you have validated the system on paper."
            )
