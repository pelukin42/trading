"""Línea de comandos: backtest, run (un paso diario de paper trading) y report."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from papertrade.alpaca import AlpacaPaperBroker, BrokerError
from papertrade.config import Config, ConfigError, load_config, load_env_file
from papertrade.data import DataError, MarketData, alpaca_credentials, load_market_data
from papertrade.engine import (
    SimResult,
    backtest_decider,
    current_weights,
    decide_step,
    plan_orders,
    simulate,
)
from papertrade.ledger import Ledger, decision_from_entry
from papertrade.metrics import ReportInput, build_report
from papertrade.strategies import Decision, Strategy, build_strategy

LLM_BACKTEST_REFUSAL = """\
Esta estrategia usa un LLM y un backtest con LLM está contaminado: el modelo se
entrenó con noticias y precios posteriores a esas fechas, así que "conoce el futuro".
Además, cada día simulado cuesta llamadas de API.

La única evaluación válida para un agente LLM es paper trading hacia adelante:
  papertrade run <config>   (una vez por día hábil, después del cierre)

Si igual quieres correrlo, por ejemplo para probar que la integración funciona,
agrega --allow-llm."""

LLM_CONTAMINATION = (
    "La estrategia usa un LLM: este backtest está contaminado por look-ahead "
    "(el modelo ya conoce lo que pasó). No lo uses para decidir si invertir."
)


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Fecha inválida (usa AAAA-MM-DD): {text}") from None


def _fmt_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{s} {w:.1%}" for s, w in weights.items()) or "todo en efectivo"


def _fmt_positions(positions: dict[str, float]) -> str:
    return ", ".join(f"{s} {q:g}" for s, q in sorted(positions.items())) or "ninguna"


def _llm_costs(entries: list[dict]) -> pd.Series:
    if not entries:
        return pd.Series(dtype=float)
    return pd.Series(
        [float(e.get("llm_cost_usd", 0.0)) for e in entries],
        index=pd.DatetimeIndex([pd.Timestamp(e["date"]) for e in entries]),
    )


def _universe_close(equity: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    cols = [f"close_{s}" for s in cfg.symbols]
    if not all(c in equity.columns for c in cols):
        return None
    return equity[cols].rename(columns=lambda c: c.removeprefix("close_"))


def _report_input(cfg: Config, title: str, equity: pd.DataFrame, fills: pd.DataFrame, entries, warnings=()):
    return ReportInput(
        title=title,
        equity=equity["equity"],
        benchmark_close=equity["benchmark_close"],
        benchmark_name=cfg.benchmark,
        n_trades=len(fills),
        commissions=float(fills["commission"].sum()) if len(fills) else 0.0,
        slippage=float(fills["slippage_cost"].sum()) if len(fills) else 0.0,
        llm_costs=_llm_costs(entries),
        universe_close=_universe_close(equity, cfg),
        risk_free_rate=cfg.risk_free_rate,
        warnings=tuple(warnings),
    )


def _save_sim(ledger: Ledger, result: SimResult, data: MarketData, cfg: Config) -> pd.DataFrame:
    equity = result.equity.copy()
    equity["benchmark_close"] = data.close[cfg.benchmark].reindex(equity.index)
    for symbol in cfg.symbols:
        equity[f"close_{symbol}"] = data.close[symbol].reindex(equity.index)
    ledger.write_equity(equity)
    ledger.write_fills(result.fills_frame())
    return equity


def cmd_backtest(cfg: Config, args) -> int:
    strategy = build_strategy(cfg.strategy, cfg.symbols)
    if strategy.uses_llm and not args.allow_llm:
        print(LLM_BACKTEST_REFUSAL, file=sys.stderr)
        return 2

    start = pd.Timestamp(args.start)
    fetch_start = args.start - timedelta(days=cfg.data.lookback_days)
    data = load_market_data(cfg.data, cfg.data_symbols, fetch_start, args.end)
    if not (data.dates >= start).any():
        raise DataError(f"No hay datos desde {args.start}")

    result = simulate(cfg, data, start, backtest_decider(strategy, cfg))
    ledger = Ledger(cfg.run_dir / "backtest")
    ledger.reset()
    equity = _save_sim(ledger, result, data, cfg)
    for day, decision in result.decisions:
        ledger.append_decision(day, decision)

    warnings = ["Es un backtest: el rendimiento pasado no garantiza resultados futuros."]
    if strategy.uses_llm:
        warnings.insert(0, LLM_CONTAMINATION)
    text, _ = build_report(
        _report_input(cfg, f"Reporte: {cfg.name} | backtest", equity, result.fills_frame(), ledger.read_decisions(), warnings)
    )
    print(text)
    print(f"\nArchivos guardados en {ledger.root}/")
    return 0


def _replay_sim(cfg: Config, ledger: Ledger, strategy: Strategy | None):
    """Reconstruye el portafolio simulado a partir de las decisiones registradas.

    Con `strategy`, además consulta al agente para el último día disponible.
    """
    recorded = {pd.Timestamp(e["date"]): decision_from_entry(e) for e in ledger.read_decisions()}
    first = min(recorded) if recorded else None
    if first is None and strategy is None:
        raise ConfigError("Todavía no hay decisiones registradas. Corre primero: papertrade run <config>")

    fetch_start = (first.date() if first is not None else date.today()) - timedelta(days=cfg.data.lookback_days)
    traded = {s for d in recorded.values() for s in d.weights}
    symbols = list(dict.fromkeys([*cfg.data_symbols, *sorted(traded)]))
    data = load_market_data(cfg.data, symbols, fetch_start)
    latest = data.dates[-1]

    orphan = sorted(d for d in recorded if d not in data.dates)
    if orphan:
        print(f"AVISO: {len(orphan)} decisiones con fechas sin datos de precio (p. ej. {orphan[0]:%Y-%m-%d}); se ignoran.")

    last_decision = max(recorded) if recorded else None
    new: list[tuple[pd.Timestamp, Decision]] = []

    def decide(day, data_, current, drawdown):
        if day in recorded:
            return recorded[day]
        if strategy is not None and day == latest:
            decision = decide_step(strategy, cfg, day, data_, current, drawdown, last_decision)
            if decision is not None:
                new.append((day, decision))
            return decision
        return None

    start = first if first is not None else latest
    result = simulate(cfg, data, min(start, latest), decide)
    return result, data, latest, new, latest in recorded


def _run_sim(cfg: Config, strategy: Strategy, ledger: Ledger) -> int:
    result, data, latest, new, already = _replay_sim(cfg, ledger, strategy)
    for day, decision in new:
        ledger.append_decision(day, decision)
    _save_sim(ledger, result, data, cfg)

    last = result.equity.iloc[-1]
    print(f"[{cfg.name}] {latest:%Y-%m-%d} | broker simulado")
    print(f"Valor del portafolio: ${last['equity']:,.2f} (efectivo ${last['cash']:,.2f})")
    print(f"Posiciones: {_fmt_positions(result.positions)}")
    if already:
        print("Ya había una decisión registrada para este día; no se consultó al agente de nuevo.")
    elif new:
        decision = new[-1][1]
        print(f"Nueva decisión: {_fmt_weights(decision.weights)}")
        if decision.notes:
            print(f"Notas: {decision.notes}")
    else:
        print(f"Hoy no toca rebalancear (frecuencia: {cfg.rebalance}).")
    if result.pending:
        orders = ", ".join(f"{'COMPRAR' if o.side == 'buy' else 'VENDER'} {o.qty:g} {o.symbol}" for o in result.pending)
        print(f"Órdenes para la próxima apertura: {orders}")
    print(f"\nVer el reporte: papertrade report <config>   (archivos en {ledger.root}/)")
    return 0


def _run_alpaca(cfg: Config, strategy: Strategy, ledger: Ledger) -> int:
    key, secret = alpaca_credentials()
    broker = AlpacaPaperBroker(key, secret)
    history = ledger.read_equity()
    entries = ledger.read_decisions()

    data = load_market_data(cfg.data, cfg.data_symbols, date.today() - timedelta(days=cfg.data.lookback_days))
    latest = data.dates[-1]
    if latest in history.index:
        print(f"Ya se registró el día {latest:%Y-%m-%d}. Vuelve a correrlo el próximo día hábil.")
        return 0

    snap = broker.account()
    ledger.append_equity(latest, snap.equity, snap.cash, float(data.close[cfg.benchmark].iloc[-1]))
    equity_series = pd.concat([history["equity"].astype(float), pd.Series([snap.equity], index=[latest])])
    drawdown = 1 - snap.equity / equity_series.max() if equity_series.max() > 0 else 0.0

    prices = {**snap.prices, **data.close.iloc[-1].to_dict()}
    current = current_weights(snap.positions, prices, snap.equity)
    last_decision = pd.Timestamp(entries[-1]["date"]) if entries else None

    print(f"[{cfg.name}] {latest:%Y-%m-%d} | Alpaca PAPER")
    print(f"Valor de la cuenta: ${snap.equity:,.2f} (efectivo ${snap.cash:,.2f})")
    print(f"Posiciones: {_fmt_positions(snap.positions)}")

    decision = decide_step(strategy, cfg, latest, data, current, drawdown, last_decision)
    if decision is None:
        print(f"Hoy no toca rebalancear (frecuencia: {cfg.rebalance}).")
        return 0

    ledger.append_decision(latest, decision)
    print(f"Nueva decisión: {_fmt_weights(decision.weights)}")
    orders = plan_orders(decision.weights, snap.positions, prices, snap.equity, cfg.risk.rebalance_band, cfg.broker.fractional)
    broker.cancel_open_orders()
    for order in orders:
        resp = broker.submit(order)
        ledger.append_order(
            {
                "date": latest.strftime("%Y-%m-%d"),
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "alpaca_id": resp.get("id", ""),
                "status": resp.get("status", ""),
            }
        )
        print(f"Orden enviada: {order.side} {order.qty:g} {order.symbol} ({resp.get('status', '?')})")
    if not orders:
        print("La decisión no requiere órdenes (cambios menores a la banda de rebalanceo).")
    return 0


def cmd_run(cfg: Config, args) -> int:
    strategy = build_strategy(cfg.strategy, cfg.symbols)
    ledger = Ledger(cfg.run_dir)
    if cfg.broker.type == "alpaca":
        return _run_alpaca(cfg, strategy, ledger)
    return _run_sim(cfg, strategy, ledger)


def cmd_report(cfg: Config, args) -> int:
    if args.backtest:
        ledger = Ledger(cfg.run_dir / "backtest")
        equity = ledger.read_equity()
        if equity.empty:
            raise ConfigError("No hay backtest guardado. Corre primero: papertrade backtest <config> --start AAAA-MM-DD")
        strategy = build_strategy(cfg.strategy, cfg.symbols)
        warnings = [LLM_CONTAMINATION] if strategy.uses_llm else []
        title = f"Reporte: {cfg.name} | backtest"
        text, _ = build_report(_report_input(cfg, title, equity, ledger.read_fills(), ledger.read_decisions(), warnings))
        print(text)
        return 0

    ledger = Ledger(cfg.run_dir)
    if cfg.broker.type == "sim":
        result, data, *_ = _replay_sim(cfg, ledger, strategy=None)
        equity = _save_sim(ledger, result, data, cfg)
        title = f"Reporte: {cfg.name} | paper trading simulado"
        text, _ = build_report(_report_input(cfg, title, equity, result.fills_frame(), ledger.read_decisions()))
        print(text)
        return 0

    equity = ledger.read_equity()
    if equity.empty:
        raise ConfigError("Todavía no hay registros. Corre primero: papertrade run <config>")
    warnings = ["Alpaca paper no simula dividendos ni impacto de mercado; el benchmark se compara sin dividendos."]
    try:
        prices = load_market_data(cfg.data, cfg.data_symbols, equity.index[0].date(), adjust="split")
        for symbol in cfg.data_symbols:
            column = "benchmark_close" if symbol == cfg.benchmark else f"close_{symbol}"
            equity[column] = prices.close[symbol].reindex(equity.index).ffill()
        if cfg.benchmark in cfg.symbols:
            equity[f"close_{cfg.benchmark}"] = equity["benchmark_close"]
    except DataError as exc:
        warnings.append(f"No se pudieron actualizar los precios ({exc}); se usan los registrados.")
    orders = ledger.read_orders()
    fills = pd.DataFrame({"commission": [0.0] * len(orders), "slippage_cost": [0.0] * len(orders)})
    title = f"Reporte: {cfg.name} | Alpaca paper"
    text, _ = build_report(_report_input(cfg, title, equity, fills, ledger.read_decisions(), warnings))
    print(text)
    return 0


COMMANDS = {"backtest": cmd_backtest, "run": cmd_run, "report": cmd_report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papertrade",
        description="Paper trading para evaluar agentes contra un benchmark, con costos incluidos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Simula la estrategia sobre datos históricos")
    bt.add_argument("config", help="Archivo TOML en configs/")
    bt.add_argument("--start", required=True, type=_parse_date, help="Fecha de inicio AAAA-MM-DD")
    bt.add_argument("--end", type=_parse_date, help="Fecha final AAAA-MM-DD (por defecto, hoy)")
    bt.add_argument("--allow-llm", action="store_true", help="Permite backtest de estrategias con LLM (contaminado)")

    run = sub.add_parser("run", help="Ejecuta el paso del día: decide y envía órdenes (paper)")
    run.add_argument("config", help="Archivo TOML en configs/")

    rep = sub.add_parser("report", help="Muestra métricas contra el benchmark")
    rep.add_argument("config", help="Archivo TOML en configs/")
    rep.add_argument("--backtest", action="store_true", help="Reporta el último backtest en vez del paper trading")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command](cfg, args)
    except (ConfigError, DataError, BrokerError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
