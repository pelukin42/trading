"""Agent loop tests with a scripted stand-in for the Anthropic client (no network, no API key)."""

import json
from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from agentrade.agent import TradingAgent
from agentrade.broker.sim import SimBroker
from agentrade.config import Settings
from agentrade.data import synthetic_history
from agentrade.journal import Journal
from agentrade.risk import RiskManager
from agentrade.router import OrderRouter


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def response(content, stop_reason, **extra):
    usage = NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500, cache_creation_input_tokens=0)
    return NS(content=content, stop_reason=stop_reason, usage=usage, **extra)


class ScriptedClient:
    """Mimics client.beta.messages.create, replaying canned responses and recording requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def stack(tmp_path):
    history = synthetic_history(["AAA", "BBB"], days=260, seed=1)
    broker = SimBroker(history, initial_cash=100_000, start_index=250)
    journal = Journal(":memory:")
    settings = Settings(watchlist=["AAA", "BBB"], data_dir=tmp_path, max_agent_steps=6)
    risk = RiskManager(settings.risk, allowed_symbols=settings.watchlist, kill_switch_path=settings.kill_switch_path)
    router = OrderRouter(broker, risk, journal)
    return NS(broker=broker, journal=journal, settings=settings, router=router)


def make_agent(stack, responses):
    client = ScriptedClient(responses)
    return TradingAgent(client, stack.broker, stack.router, stack.journal, stack.settings), client


def test_full_cycle_risk_rejection_then_valid_order(stack):
    price = stack.broker.get_latest_price("AAA")
    agent, client = make_agent(stack, [
        response([tool_use("t1", "get_portfolio", {}), tool_use("t2", "get_market_data", {"symbols": ["AAA", "BBB"]})],
                 "tool_use"),
        # Oversized and without a stop: the risk engine must reject it.
        response([tool_use("t3", "place_order", {"symbol": "AAA", "side": "buy", "quantity": 50_000,
                                                 "quantity_type": "usd", "order_type": "market",
                                                 "rationale": "all in"})], "tool_use"),
        response([tool_use("t4", "place_order", {"symbol": "AAA", "side": "buy", "quantity": 10_000,
                                                 "quantity_type": "usd", "order_type": "market",
                                                 "stop_loss_price": round(price * 0.92, 2),
                                                 "rationale": "trend intact, sized to 10%"})], "tool_use"),
        response([text("Compré AAA con stop al 8%.")], "end_turn"),
    ])
    result = agent.run_cycle()

    assert result.status == "completed"
    assert result.report == "Compré AAA con stop al 8%."
    assert result.steps == 4

    # Parallel tool calls are answered in a single user message, in order.
    second_request = client.calls[1]["messages"]
    results = second_request[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    snapshots = json.loads(results[1]["content"])["snapshots"]
    assert {s["symbol"] for s in snapshots} == {"AAA", "BBB"}

    # The rejection reasons are fed back to the model.
    rejection = json.loads(client.calls[2]["messages"][-1]["content"][0]["content"])
    assert rejection["approved_by_risk"] is False
    assert any("stop_loss_price is required" in r for r in rejection["risk_rejections"])
    assert any("of equity" in r for r in rejection["risk_rejections"])

    # Only the valid order reached the broker; both are in the journal with their rationale.
    pending = stack.broker.get_open_orders()
    assert len(pending) == 1 and pending[0].request.qty == 10_000 // price
    rows = stack.journal.orders_for_cycle(result.cycle_id)
    assert [r["approved"] for r in rows] == [0, 1]
    assert rows[1]["rationale"] == "trend intact, sized to 10%"

    cycle = stack.journal.recent_cycles(1)[0]
    assert cycle["status"] == "completed" and cycle["input_tokens"] == 4000


def test_request_shape(stack):
    agent, client = make_agent(stack, [response([text("Sin cambios.")], "end_turn")])
    agent.run_cycle()
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["thinking"]["type"] == "adaptive"
    assert req["output_config"] == {"effort": "medium"}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in req["tools"])
    assert "AAA, BBB" in req["messages"][0]["content"]
    assert "español" in req["messages"][0]["content"]


def test_previous_report_is_carried_into_next_cycle(stack):
    agent, client = make_agent(stack, [response([text("Tesis: AAA en tendencia.")], "end_turn"),
                                       response([text("ok")], "end_turn")])
    agent.run_cycle()
    agent.run_cycle()
    assert "Tesis: AAA en tendencia." in client.calls[1]["messages"][0]["content"]


def test_refusal_and_api_errors_end_the_cycle_cleanly(stack):
    agent, _ = make_agent(stack, [response([], "refusal", stop_details=NS(category="cyber"))])
    assert agent.run_cycle().status == "refused"

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.InternalServerError("boom", response=httpx2.Response(500, request=req), body=None)
    agent, _ = make_agent(stack, [err])
    result = agent.run_cycle()
    assert result.status == "api_error" and "500" in result.report


def test_step_limit_and_tool_errors(stack):
    loop = [response([tool_use(f"t{i}", "get_price_history", {"symbol": "ZZZ", "days": 30})], "tool_use")
            for i in range(10)]
    agent, client = make_agent(stack, loop)
    result = agent.run_cycle()
    assert result.status == "step_limit" and result.steps == stack.settings.max_agent_steps
    last_result = client.calls[-1]["messages"][-1]["content"][0]
    assert last_result["is_error"] is True and "ZZZ" in last_result["content"]


def test_kill_switch_skips_cycle_without_calling_the_model(stack):
    stack.settings.kill_switch_path.write_text("stop")
    agent, client = make_agent(stack, [])
    assert agent.run_cycle().status == "halted"
    assert client.calls == []


def test_agent_inside_backtester_over_several_cycles(tmp_path):
    from agentrade.backtest import Backtester
    from agentrade.config import RiskLimits

    history = synthetic_history(["AAA", "BBB"], days=240, seed=2)
    bt = Backtester(history, RiskLimits())
    broker, router = bt.make_stack(warmup=200)
    settings = Settings(watchlist=["AAA", "BBB"], data_dir=tmp_path, mode="backtest")
    price = broker.get_latest_price("AAA")
    client = ScriptedClient([
        response([tool_use("b1", "place_order", {"symbol": "AAA", "side": "buy", "quantity": 10_000,
                                                 "quantity_type": "usd", "order_type": "market",
                                                 "stop_loss_price": round(price * 0.88, 2),
                                                 "rationale": "entry"})], "tool_use"),
        response([text("Entré en AAA.")], "end_turn"),
        response([text("Mantengo.")], "end_turn"),
        response([text("Mantengo.")], "end_turn"),
    ])
    agent = TradingAgent(client, broker, router, bt.journal, settings)

    res = bt.run(lambda b, r: agent.run_cycle(), warmup=200, every=5, max_decisions=3, broker=broker, router=router)

    assert res.decisions == 3
    entry = res.fills[0]
    assert entry.symbol == "AAA" and entry.side.value == "buy" and entry.t > history["AAA"][200].t
    assert all(f.reason == "stop" for f in res.fills[1:])  # only the stop can exit; the agent never sold
    # The second cycle saw the fill and the first report.
    assert "Entré en AAA." in client.calls[2]["messages"][0]["content"]


def test_close_position_sells_everything(stack):
    agent, _ = make_agent(stack, [])
    from agentrade.models import OrderRequest, Side
    price = stack.broker.get_latest_price("BBB")
    stack.router.route(OrderRequest("BBB", Side.BUY, 20, stop_loss_price=round(price * 0.9, 2)))
    stack.broker.advance()
    out, is_error = agent.executor.execute("close_position", {"symbol": "BBB", "rationale": "thesis broken"})
    assert not is_error and json.loads(out)["approved_by_risk"] is True
    stack.broker.advance()
    assert "BBB" not in stack.broker.get_positions()
