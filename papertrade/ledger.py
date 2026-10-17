"""Bitácora en disco de cada experimento (carpeta runs/<name>/).

decisions.jsonl es el registro permanente de lo que decidió el agente y cuándo
(`recorded_at`); no se reescribe. El resto de archivos se puede regenerar.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from papertrade.strategies import Decision


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def decisions_path(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def equity_path(self) -> Path:
        return self.root / "equity.csv"

    @property
    def fills_path(self) -> Path:
        return self.root / "fills.csv"

    @property
    def orders_path(self) -> Path:
        return self.root / "orders.csv"

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def read_decisions(self) -> list[dict]:
        if not self.decisions_path.is_file():
            return []
        lines = self.decisions_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append_decision(self, day: pd.Timestamp, decision: Decision) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "date": day.strftime("%Y-%m-%d"),
            "weights": {s: round(w, 6) for s, w in decision.weights.items()},
            "notes": decision.notes,
            "llm_cost_usd": decision.llm_cost_usd,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_equity(self) -> pd.DataFrame:
        if not self.equity_path.is_file():
            return pd.DataFrame(columns=["equity", "cash", "benchmark_close"], index=pd.DatetimeIndex([], name="date"))
        return pd.read_csv(self.equity_path, index_col="date", parse_dates=["date"])

    def write_equity(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.equity_path, index_label="date", date_format="%Y-%m-%d", float_format="%.4f")

    def append_equity(self, day: pd.Timestamp, equity: float, cash: float, benchmark_close: float) -> None:
        row = {"date": day.strftime("%Y-%m-%d"), "equity": equity, "cash": cash, "benchmark_close": benchmark_close}
        self._append_csv(self.equity_path, row)

    def write_fills(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.fills_path, index=False, float_format="%.4f")

    def read_fills(self) -> pd.DataFrame:
        if not self.fills_path.is_file():
            return pd.DataFrame(columns=["date", "symbol", "side", "qty", "price", "commission", "slippage_cost"])
        return pd.read_csv(self.fills_path)

    def append_order(self, row: dict) -> None:
        self._append_csv(self.orders_path, row)

    def read_orders(self) -> pd.DataFrame:
        if not self.orders_path.is_file():
            return pd.DataFrame()
        return pd.read_csv(self.orders_path)

    def _append_csv(self, path: Path, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        new = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


def decision_from_entry(entry: dict) -> Decision:
    return Decision(entry["weights"], entry.get("notes", {}), entry.get("llm_cost_usd", 0.0))
