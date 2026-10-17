"""Historical data sources for backtests: synthetic, CSV files, or a broker's market data API."""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Bar


def synthetic_history(symbols: list[str], days: int = 750, seed: int = 7,
                      start: datetime | None = None) -> dict[str, list[Bar]]:
    """Correlated random walks with per-symbol drift and volatility, on weekdays.

    Useful to exercise the full pipeline offline and, for the agent, to test decision-making on
    prices the model can't have memorized.
    """
    rng = random.Random(seed)
    start = start or datetime(2022, 1, 3, 21, tzinfo=timezone.utc)
    dates: list[datetime] = []
    d = start
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    market = [rng.gauss(0.0002, 0.011) for _ in dates]
    out: dict[str, list[Bar]] = {}
    for sym in symbols:
        drift = rng.gauss(0.0, 0.15) / 252
        vol = rng.uniform(0.18, 0.45) / math.sqrt(252)
        beta = rng.uniform(0.6, 1.4)
        price = rng.uniform(40, 400)
        bars = []
        for t, m in zip(dates, market):
            open_ = price * math.exp(rng.gauss(0, vol * 0.3))
            ret = drift + beta * m + rng.gauss(0, vol)
            close = open_ * math.exp(ret)
            hi = max(open_, close) * math.exp(abs(rng.gauss(0, vol * 0.5)))
            lo = min(open_, close) * math.exp(-abs(rng.gauss(0, vol * 0.5)))
            bars.append(Bar(t, round(open_, 2), round(hi, 2), round(lo, 2), round(close, 2),
                            float(int(rng.uniform(0.5, 1.5) * 2_000_000))))
            price = close
        out[sym.upper()] = bars
    return out


def load_csv_dir(directory: str | Path, symbols: list[str] | None = None) -> dict[str, list[Bar]]:
    """Load <SYMBOL>.csv files with columns date,open,high,low,close[,volume]."""
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    out: dict[str, list[Bar]] = {}
    wanted = {s.upper() for s in symbols} if symbols else None
    for f in files:
        sym = f.stem.upper()
        if wanted and sym not in wanted:
            continue
        bars = []
        with f.open(newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip().lower(): v for k, v in row.items()}
                t = datetime.fromisoformat(row.get("date") or row["timestamp"])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                bars.append(Bar(t, float(row["open"]), float(row["high"]), float(row["low"]),
                                float(row["close"]), float(row.get("volume") or 0)))
        out[sym] = sorted(bars, key=lambda b: b.t)
    if wanted and (missing := wanted - out.keys()):
        raise FileNotFoundError(f"No CSV for: {', '.join(sorted(missing))} in {directory}")
    return out


def yahoo_history(symbols: list[str], days: int = 750) -> dict[str, list[Bar]]:
    """Real daily bars from Yahoo Finance (no key needed; personal use)."""
    from .yahoo import YahooData

    yd = YahooData()
    return {s.upper(): yd.get_bars(s, limit=days) for s in symbols}
