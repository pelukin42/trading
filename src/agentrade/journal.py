"""SQLite audit trail: every agent cycle, tool call, risk verdict and order, with its rationale."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    model TEXT,
    mode TEXT,
    status TEXT,
    steps INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    report TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    stop_loss_price REAL,
    approved INTEGER NOT NULL,
    risk_reasons TEXT,
    broker_order_id TEXT,
    status TEXT,
    rationale TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL
);
"""


def _iso(t: datetime | None = None) -> str:
    return (t or datetime.now(timezone.utc)).isoformat()


class Journal:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def start_cycle(self, model: str, mode: str, t: datetime | None = None) -> str:
        cycle_id = uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO cycles (id, started_at, model, mode, status) VALUES (?, ?, ?, ?, 'running')",
                        (cycle_id, _iso(t), model, mode))
        self.db.commit()
        return cycle_id

    def end_cycle(self, cycle_id: str, status: str, steps: int, usage: dict[str, int], report: str,
                  t: datetime | None = None) -> None:
        self.db.execute(
            "UPDATE cycles SET ended_at=?, status=?, steps=?, input_tokens=?, output_tokens=?, "
            "cache_read_tokens=?, cache_write_tokens=?, report=? WHERE id=?",
            (_iso(t), status, steps, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
             usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0),
             report, cycle_id),
        )
        self.db.commit()

    def event(self, cycle_id: str | None, kind: str, payload: Any, t: datetime | None = None) -> None:
        self.db.execute("INSERT INTO events (cycle_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
                        (cycle_id, _iso(t), kind, json.dumps(payload, default=str, ensure_ascii=False)))
        self.db.commit()

    def order(self, cycle_id: str | None, *, symbol: str, side: str, qty: float, order_type: str,
              limit_price: float | None, stop_loss_price: float | None, approved: bool,
              risk_reasons: list[str], broker_order_id: str | None, status: str, rationale: str,
              t: datetime | None = None) -> None:
        self.db.execute(
            "INSERT INTO orders (cycle_id, ts, symbol, side, qty, order_type, limit_price, stop_loss_price, "
            "approved, risk_reasons, broker_order_id, status, rationale) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cycle_id, _iso(t), symbol, side, qty, order_type, limit_price, stop_loss_price, int(approved),
             json.dumps(risk_reasons, ensure_ascii=False), broker_order_id, status, rationale),
        )
        self.db.commit()

    def snapshot_equity(self, equity: float, cash: float, t: datetime | None = None) -> None:
        self.db.execute("INSERT INTO equity (ts, equity, cash) VALUES (?, ?, ?)", (_iso(t), equity, cash))
        self.db.commit()

    def approved_orders_since(self, since: datetime) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM orders WHERE approved=1 AND ts >= ?", (_iso(since),)).fetchone()
        return int(row[0])

    def cycles_since(self, since: datetime) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM cycles WHERE started_at >= ?", (_iso(since),)).fetchone()
        return int(row[0])

    def recent_cycles(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cycles ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()

    def orders_for_cycle(self, cycle_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM orders WHERE cycle_id=? ORDER BY id", (cycle_id,)).fetchall()

    def recent_reports(self, limit: int = 3) -> list[sqlite3.Row]:
        """Previous cycle reports, fed back to the agent so it keeps a consistent thesis across cycles."""
        return self.db.execute(
            "SELECT started_at, report FROM cycles WHERE status='completed' AND report != '' "
            "ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
