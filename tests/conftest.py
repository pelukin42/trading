from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_prices(n=300, start="2023-01-02", drift=0.0005, seed=0, base=100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = base * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": open_, "close": close})


@pytest.fixture
def csv_dir(tmp_path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    for i, symbol in enumerate(["AAA", "BBB", "SPY"]):
        make_prices(seed=i, drift=0.0003 * (i + 1)).to_csv(directory / f"{symbol}.csv", index=False)
    return directory


@pytest.fixture
def write_config(tmp_path, csv_dir):
    def _write(strategy='type = "buy_and_hold"', broker="", risk="", root="") -> Path:
        path = tmp_path / "config.toml"
        path.write_text(
            f"""
name = "test"
symbols = ["AAA", "BBB"]
benchmark = "SPY"
rebalance = "weekly"
runs_dir = "{(tmp_path / 'runs').as_posix()}"
{root}

[strategy]
{strategy}

[broker]
type = "sim"
starting_cash = 10000
slippage_bps = 10
commission_per_order = 1.0
{broker}

[data]
source = "csv"
csv_dir = "{csv_dir.as_posix()}"
lookback_days = 20000

[risk]
{risk}
""",
            encoding="utf-8",
        )
        return path

    return _write
