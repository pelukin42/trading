"""Precios diarios (apertura y cierre) desde yfinance, Alpaca o archivos CSV."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from papertrade.config import DataConfig

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


class DataError(RuntimeError):
    pass


@dataclass
class MarketData:
    """Precios alineados: filas = días hábiles, columnas = tickers."""

    open: pd.DataFrame
    close: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def until(self, day: pd.Timestamp) -> MarketData:
        """Solo los datos conocidos al cierre de `day` (sin mirar al futuro)."""
        return MarketData(self.open.loc[:day], self.close.loc[:day])

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> MarketData:
        missing = [s for s, f in frames.items() if f.empty]
        if missing:
            raise DataError(f"Sin datos para: {', '.join(missing)}")
        frames = {s: f[~f.index.duplicated(keep="last")] for s, f in frames.items()}
        close =pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index().ffill()
        open_ = pd.DataFrame({s: f["open"] for s, f in frames.items()}).sort_index()
        open_ = open_.reindex(close.index).fillna(close)
        complete = close.notna().all(axis=1)
        if not complete.any():
            raise DataError("Los tickers no tienen fechas en común")
        first = complete.idxmax()
        close, open_ = close.loc[first:], open_.loc[first:]
        close.index.name = open_.index.name = "date"
        return cls(open=open_.astype(float), close=close.astype(float))


def load_market_data(
    cfg: DataConfig,
    symbols: list[str],
    start: date,
    end: date | None = None,
    adjust: str = "all",
    session: requests.Session | None = None,
) -> MarketData:
    """`adjust="all"` ajusta por splits y dividendos; `"split"` solo por splits."""
    if cfg.source == "csv":
        frames = _load_csv(Path(cfg.csv_dir), symbols)
    elif cfg.source == "yfinance":
        frames = _load_yfinance(symbols, start, end, adjust)
    elif cfg.source == "alpaca":
        frames = _load_alpaca(symbols, start, end, adjust, cfg.feed, session or requests.Session())
    else:
        raise DataError(f"Fuente de datos desconocida: {cfg.source}")

    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else None
    frames = {s: f.loc[lo:hi] for s, f in frames.items()}
    return MarketData.from_frames(frames)


def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _load_csv(directory: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        path = directory / f"{symbol}.csv"
        if not path.is_file():
            raise DataError(f"No existe {path} (columnas requeridas: date, open, close)")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not {"date", "open", "close"} <= set(df.columns):
            raise DataError(f"{path} debe tener las columnas date, open y close")
        df.index = _naive_dates(pd.to_datetime(df.pop("date")))
        frames[symbol] = df[["open", "close"]].sort_index()
    return frames


def _load_yfinance(symbols, start: date, end: date | None, adjust: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        raise DataError("Falta yfinance. Instálalo con: pip install -e '.[yfinance]'") from None

    frames = {}
    for symbol in symbols:
        hist = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat() if end else None,
            auto_adjust=(adjust == "all"),
            actions=False,
        )
        if hist.empty:
            raise DataError(f"yfinance no devolvió datos para {symbol}")
        frames[symbol] = pd.DataFrame(
            {"open": hist["Open"].to_numpy(), "close": hist["Close"].to_numpy()},
            index=_naive_dates(hist.index),
        )
    return frames


def alpaca_credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise DataError(
            "Faltan ALPACA_API_KEY y ALPACA_SECRET_KEY. Cópialas de tu cuenta paper de Alpaca a .env"
        )
    return key, secret


def _load_alpaca(symbols, start: date, end: date | None, adjust: str, feed: str, session) -> dict[str, pd.DataFrame]:
    key, secret = alpaca_credentials()
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "adjustment": adjust,
        "feed": feed,
        "limit": 10000,
        "sort": "asc",
    }
    if end:
        params["end"] = (end + timedelta(days=1)).isoformat()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    rows: dict[str, list[dict]] = {s: [] for s in symbols}
    while True:
        resp = session.get(ALPACA_DATA_URL, params=params, headers=headers, timeout=30)
        if resp.status_code >= 400:
            raise DataError(f"Alpaca datos → HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        for symbol, bars in (payload.get("bars") or {}).items():
            rows.setdefault(symbol, []).extend(bars)
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    frames = {}
    for symbol in symbols:
        bars = rows.get(symbol) or []
        if not bars:
            raise DataError(f"Alpaca no devolvió datos para {symbol}")
        stamps = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert("America/New_York")
        frames[symbol] = pd.DataFrame(
            {"open": [b["o"] for b in bars], "close": [b["c"] for b in bars]},
            index=_naive_dates(stamps),
        )
    return frames
