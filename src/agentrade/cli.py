"""Command-line interface: backtest, run, status, kill/resume, journal."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .agent import CycleResult
from .backtest import Backtester, BacktestResult
from .broker.base import BrokerError
from .config import ConfigError, Settings, load_dotenv
from .data import load_csv_dir, synthetic_history, yahoo_history
from .engine import build_agent, build_stack, run_loop
from .strategies import STRATEGIES


def _print_table(rows: list[tuple[str, str]], right: list[tuple[str, str]] | None = None,
                 headers: tuple[str, str, str] = ("", "Estrategia", "Buy & hold")) -> None:
    w = max(len(r[0]) for r in rows) + 2
    print(f"{headers[0]:<{w}}{headers[1]:>14}{headers[2]:>14}")
    for i, (k, v) in enumerate(rows):
        other = right[i][1] if right else ""
        print(f"{k:<{w}}{v:>14}{other:>14}")


def _print_cycle(result: CycleResult, model: str) -> None:
    cost = result.estimated_cost(model)
    print(f"\n== Ciclo {result.cycle_id}: {result.status} ({result.steps} pasos) ==")
    for o in result.orders:
        print("  orden:", o["result"])
    print(result.report or "(sin reporte)")
    u = result.usage
    if u:
        print(f"\ntokens: entrada={u.get('input_tokens', 0)} salida={u.get('output_tokens', 0)} "
              f"cache_lectura={u.get('cache_read_input_tokens', 0)}"
              + (f"  costo estimado ≈ ${cost:.3f}" if cost is not None else ""))


def _load_history(args: argparse.Namespace, settings: Settings):
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else settings.watchlist
    if args.source == "synthetic":
        return synthetic_history(symbols, days=args.days, seed=args.seed)
    if args.source == "yahoo":
        return yahoo_history(symbols, days=args.days)
    if args.source.startswith("csv:"):
        return load_csv_dir(args.source[4:], symbols)
    if args.source == "alpaca":
        from .broker.alpaca import AlpacaBroker
        broker = AlpacaBroker(settings.alpaca_key_id, settings.alpaca_secret_key, data_feed=settings.alpaca_data_feed)
        return {s: broker.get_bars(s, limit=args.days) for s in symbols}
    raise SystemExit(f"Fuente desconocida: {args.source}")


def _report_backtest(name: str, res: BacktestResult) -> None:
    print(f"\nBacktest: {name} — {res.decisions} decisiones, {res.rejected_orders} órdenes rechazadas por riesgo")
    _print_table(res.metrics.as_rows(), res.benchmark.as_rows())


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    history = _load_history(args, settings)
    if args.strategy == "agent":
        return _backtest_agent(args, settings, history)
    strategy = STRATEGIES[args.strategy]()
    bt = Backtester(history, settings.risk, initial_cash=args.cash)
    _report_backtest(strategy.name, bt.run_strategy(strategy, every=args.every))
    return 0


def _backtest_agent(args: argparse.Namespace, settings: Settings, history) -> int:
    import anthropic

    from .agent import TradingAgent

    settings.watchlist = list(history)
    settings.mode = "backtest"
    settings.enable_web_search = False  # news from "today" would leak the future into the past
    bt = Backtester(history, settings.risk, initial_cash=args.cash)
    warmup = 200
    broker, router = bt.make_stack(warmup)
    agent = TradingAgent(anthropic.Anthropic(), broker, router, bt.journal, settings)
    total_cost = 0.0

    def decide(_broker, _router) -> None:
        nonlocal total_cost
        result = agent.run_cycle()
        total_cost += result.estimated_cost(settings.model) or 0.0
        print(f"[{_broker.now.date()}] ciclo {result.status}, {len(result.orders)} órdenes, "
              f"costo acumulado ≈ ${total_cost:.2f}")

    res = bt.run(decide, warmup=warmup, every=args.every, max_decisions=args.max_cycles, broker=broker, router=router)
    _report_backtest(f"agente ({settings.model})", res)
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    stack = build_stack(settings)
    agent = build_agent(stack)
    print(f"agentrade en modo {settings.mode.upper()} | modelo {settings.model} | "
          f"watchlist {', '.join(settings.watchlist)}")
    run_loop(stack, agent, once=args.once, on_cycle=lambda r: _print_cycle(r, settings.model))
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    stack = build_stack(settings)
    a = stack.broker.get_account()
    clock = stack.broker.clock()
    print(f"Broker: {stack.broker.name}  Modo: {settings.mode}  "
          f"Mercado: {'ABIERTO' if clock.is_open else f'cerrado (abre {clock.next_open:%Y-%m-%d %H:%M} UTC)'}")
    print(f"Equity: ${a.equity:,.2f}  Cash: ${a.cash:,.2f}  P&L hoy: ${a.day_pnl:,.2f} ({a.day_pnl_pct:+.2%})")
    print(f"Kill switch: {'ACTIVO' if stack.risk.kill_switch_active() else 'inactivo'}")
    for p in stack.broker.get_positions().values():
        print(f"  {p.symbol:<6} {p.qty:>8g} @ {p.avg_price:>9.2f}  ahora {p.market_price:>9.2f}  "
              f"P&L {p.unrealized_pnl:>+10.2f} ({p.unrealized_pnl_pct:+.2%})")
    for o in stack.broker.get_open_orders():
        print(f"  pendiente: {o.request.side.value} {o.remaining_qty:g} {o.symbol} ({o.request.order_type.value})")
    return 0


def cmd_reset_paper(args: argparse.Namespace, settings: Settings) -> int:
    if settings.broker != "local":
        print("reset-paper solo aplica a la cuenta de práctica local (AGENTRADE_BROKER=local).")
        return 2
    if not args.yes:
        print("Esto borra posiciones, órdenes e historial de la cuenta de práctica. Repite con --yes para confirmar.")
        return 2
    from .broker.local_paper import LocalPaperBroker
    from .yahoo import YahooData
    cash = args.cash if args.cash is not None else settings.paper_initial_cash
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # fresh=True: never reads the old file, so this also recovers a damaged one.
    LocalPaperBroker(YahooData(), settings.paper_state_path, initial_cash=cash, fresh=True)
    print(f"Cuenta de práctica reiniciada con ${cash:,.2f}.")
    return 0


def cmd_kill(args: argparse.Namespace, settings: Settings) -> int:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.kill_switch_path.write_text("halted by user\n")
    print("Kill switch ACTIVADO: el agente no enviará más órdenes.")
    if args.flatten:
        stack = build_stack(settings)
        stack.broker.cancel_all_orders()
        stack.broker.close_all_positions()
        print("Órdenes canceladas y posiciones cerradas a mercado.")
    return 0


def cmd_resume(args: argparse.Namespace, settings: Settings) -> int:
    settings.kill_switch_path.unlink(missing_ok=True)
    print("Kill switch desactivado.")
    return 0


def cmd_journal(args: argparse.Namespace, settings: Settings) -> int:
    from .journal import Journal
    j = Journal(settings.journal_path)
    for c in j.recent_cycles(args.n):
        print(f"\n== {c['started_at']}  {c['id']}  {c['status']}  pasos={c['steps']} "
              f"tokens={c['input_tokens']}/{c['output_tokens']} ==")
        for o in j.orders_for_cycle(c["id"]):
            verdict = "OK" if o["approved"] else f"RECHAZADA {json.loads(o['risk_reasons'])}"
            print(f"  {o['side']} {o['qty']:g} {o['symbol']} [{verdict}] — {o['rationale']}")
        print(c["report"] or "")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agentrade", description="Plataforma de trading agéntico")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="Simular una estrategia o el agente sobre datos históricos")
    p.add_argument("--strategy", choices=[*STRATEGIES, "agent"], default="sma_cross")
    p.add_argument("--symbols", help="Lista separada por comas (por defecto, la watchlist)")
    p.add_argument("--source", default="synthetic", help="synthetic | yahoo | alpaca | csv:<directorio>")
    p.add_argument("--days", type=int, default=750)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--every", type=int, default=1, help="Decidir cada N barras (agente: usa 5 = semanal)")
    p.add_argument("--max-cycles", type=int, default=20, help="Tope de ciclos del agente (controla el costo)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("run", help="Operar con el agente (paper por defecto)")
    p.add_argument("--once", action="store_true", help="Un solo ciclo y salir (útil con cron)")
    p.set_defaults(func=cmd_run)

    sub.add_parser("status", help="Cuenta, posiciones y estado del kill switch").set_defaults(func=cmd_status)

    p = sub.add_parser("kill", help="Detener todo el trading")
    p.add_argument("--flatten", action="store_true", help="Además cancelar órdenes y cerrar posiciones")
    p.set_defaults(func=cmd_kill)

    sub.add_parser("resume", help="Desactivar el kill switch").set_defaults(func=cmd_resume)

    p = sub.add_parser("reset-paper", help="Reiniciar la cuenta de práctica local")
    p.add_argument("--cash", type=float, default=None, help="Por defecto, AGENTRADE_PAPER_CASH")
    p.add_argument("--yes", action="store_true", help="Confirmar el reinicio")
    p.set_defaults(func=cmd_reset_paper)

    p = sub.add_parser("journal", help="Ver los últimos ciclos del agente")
    p.add_argument("-n", type=int, default=5)
    p.set_defaults(func=cmd_journal)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    if not args.verbose:
        for noisy in ("httpx", "httpx2", "anthropic"):  # one line per HTTP request otherwise
            logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        print(f"Configuración inválida: {e}", file=sys.stderr)
        if args.command == "kill":
            # The emergency stop must work even with a broken config: it only needs the data dir.
            data_dir = Path(os.environ.get("AGENTRADE_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "KILL").write_text("halted by user\n")
            print("Kill switch ACTIVADO igualmente. Corrige la configuración para poder usar --flatten.")
            return 0
        return 2
    try:
        return args.func(args, settings)
    except PermissionError as e:
        print(f"Bloqueado: {e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(f"Configuración inválida: {e}", file=sys.stderr)
        return 2
    except BrokerError as e:
        print(f"Error del broker: {e}", file=sys.stderr)
        return 1
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print("Falta la credencial de Claude: define ANTHROPIC_API_KEY en .env "
              "(https://console.anthropic.com).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
