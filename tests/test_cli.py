import json
from pathlib import Path

import pandas as pd
import pytest

from papertrade import cli
from papertrade.alpaca import AccountSnapshot
from papertrade.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_are_valid():
    paths = sorted((ROOT / "configs").glob("*.toml"))
    assert paths
    for path in paths:
        load_config(path)


def test_backtest_writes_files_and_report(write_config, tmp_path, capsys):
    cfg = write_config(strategy='type = "sma_crossover"\nfast = 10\nslow = 50')
    assert cli.main(["backtest", str(cfg), "--start", "2023-03-01"]) == 0
    out = capsys.readouterr().out
    assert "Benchmark (SPY)" in out and "Veredicto:" in out and "Mismos tickers" in out

    run_dir = tmp_path / "runs" / "test" / "backtest"
    equity = pd.read_csv(run_dir / "equity.csv")
    assert equity["date"].iloc[0] >= "2023-03-01"
    assert len(pd.read_csv(run_dir / "fills.csv")) > 0
    assert (run_dir / "decisions.jsonl").read_text().strip()

    assert cli.main(["report", str(cfg), "--backtest"]) == 0
    assert "Veredicto:" in capsys.readouterr().out


def test_llm_backtest_requires_explicit_flag(write_config, capsys):
    cfg = write_config(strategy='type = "tradingagents"')
    assert cli.main(["backtest", str(cfg), "--start", "2023-03-01"]) == 2
    assert "conoce el futuro" in capsys.readouterr().err


def test_sim_run_replays_decisions_across_days(write_config, csv_dir, tmp_path, capsys):
    cfg = write_config()
    full = {s: pd.read_csv(csv_dir / f"{s}.csv") for s in ["AAA", "BBB", "SPY"]}

    def show_days(n):
        for symbol, df in full.items():
            df.iloc[:n].to_csv(csv_dir / f"{symbol}.csv", index=False)

    run_dir = tmp_path / "runs" / "test"
    show_days(250)
    assert cli.main(["run", str(cfg)]) == 0
    assert "Órdenes para la próxima apertura" in capsys.readouterr().out
    decisions = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 1 and decisions[0]["date"] == full["AAA"]["date"].iloc[249]

    show_days(260)
    assert cli.main(["run", str(cfg)]) == 0
    equity = pd.read_csv(run_dir / "equity.csv")
    fills = pd.read_csv(run_dir / "fills.csv")
    assert len(equity) == 11
    assert set(fills["symbol"]) == {"AAA", "BBB"}
    assert fills["date"].iloc[0] == full["AAA"]["date"].iloc[250]  # apertura siguiente a la decisión

    assert len((run_dir / "decisions.jsonl").read_text().splitlines()) == 2  # semana nueva: "mantener"

    capsys.readouterr()
    assert cli.main(["run", str(cfg)]) == 0  # mismo día: no vuelve a consultar al agente
    assert "Ya había una decisión" in capsys.readouterr().out
    assert len((run_dir / "decisions.jsonl").read_text().splitlines()) == 2

    assert cli.main(["report", str(cfg)]) == 0
    assert "paper trading simulado" in capsys.readouterr().out


def test_report_before_any_run_explains_next_step(write_config, capsys):
    assert cli.main(["report", str(write_config())]) == 1
    assert "papertrade run" in capsys.readouterr().err


def test_missing_required_key_is_reported(write_config, capsys):
    cfg = write_config()
    cfg.write_text(cfg.read_text().replace('name = "test"', ""))
    assert cli.main(["run", str(cfg)]) == 1
    assert "obligatoria" in capsys.readouterr().err


def test_invalid_config_key_is_reported(write_config, capsys):
    cfg = write_config(risk="max_wieght = 0.5")
    assert cli.main(["run", str(cfg)]) == 1
    assert "max_wieght" in capsys.readouterr().err


class FakeAlpaca:
    instances = []

    def __init__(self, key, secret):
        self.submitted = []
        self.cancelled = 0
        FakeAlpaca.instances.append(self)

    def account(self):
        return AccountSnapshot(cash=10_000.0, equity=10_000.0, positions={}, prices={})

    def cancel_open_orders(self):
        self.cancelled += 1

    def submit(self, order):
        self.submitted.append(order)
        return {"id": f"id-{len(self.submitted)}", "status": "accepted"}


def test_alpaca_run_submits_orders_once_per_day(write_config, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(cli, "AlpacaPaperBroker", FakeAlpaca)
    FakeAlpaca.instances.clear()
    cfg = write_config()
    cfg.write_text(cfg.read_text().replace('type = "sim"', 'type = "alpaca"'))

    assert cli.main(["run", str(cfg)]) == 0
    broker = FakeAlpaca.instances[-1]
    assert broker.cancelled == 1
    assert {o.symbol for o in broker.submitted} == {"AAA", "BBB"}
    orders = pd.read_csv(tmp_path / "runs" / "test" / "orders.csv")
    assert list(orders["alpaca_id"]) == ["id-1", "id-2"]

    assert cli.main(["run", str(cfg)]) == 0
    assert "Ya se registró" in capsys.readouterr().out
    assert FakeAlpaca.instances[-1].submitted == []

    assert cli.main(["report", str(cfg)]) == 0
    assert "Alpaca paper no simula dividendos" in capsys.readouterr().out
