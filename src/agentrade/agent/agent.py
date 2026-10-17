"""The trading agent: one cycle = Claude reviews the account and market through tools and acts.

A manual tool-use loop (rather than the SDK's beta tool runner) so every step is bounded,
journaled, and the risk engine sits in the only code path that can reach the broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from ..broker.base import Broker, BrokerError
from ..config import RiskLimits, Settings
from ..journal import Journal
from ..router import OrderRouter
from .prompts import SYSTEM_PROMPT, kickoff_message
from .tools import TOOLS, WEB_SEARCH_TOOL, ToolExecutor

MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: (input, output, cache read). Cache writes bill at 1.25x input. Estimates only.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


@dataclass
class CycleResult:
    cycle_id: str
    status: str  # completed | refused | truncated | step_limit | api_error | broker_error | halted
    steps: int
    report: str
    usage: dict[str, int] = field(default_factory=dict)
    orders: list[dict[str, Any]] = field(default_factory=list)

    def estimated_cost(self, model: str) -> float | None:
        if model not in PRICING:
            return None
        inp, out, cache_read = PRICING[model]
        u = self.usage
        return (u.get("input_tokens", 0) * inp + u.get("output_tokens", 0) * out
                + u.get("cache_read_input_tokens", 0) * cache_read
                + u.get("cache_creation_input_tokens", 0) * inp * 1.25) / 1_000_000


def limits_summary(L: RiskLimits) -> str:
    return (f"max {L.max_position_pct:.0%} of equity per symbol; max gross exposure {L.max_gross_exposure_pct:.0%}; "
            f"max {L.max_open_positions} open positions; max ${L.max_order_notional:,.0f} per order; "
            f"new risk blocked after a {L.max_daily_loss_pct:.0%} daily loss; max {L.max_orders_per_day} orders/day; "
            f"stop loss {'required' if L.require_stop_loss else 'optional'} on buys "
            f"(at most {L.max_stop_distance_pct:.0%} below entry); "
            f"short selling {'allowed' if L.allow_short else 'disabled'}")


class TradingAgent:
    def __init__(self, client: anthropic.Anthropic, broker: Broker, router: OrderRouter,
                 journal: Journal, settings: Settings):
        self.client = client
        self.broker = broker
        self.router = router
        self.journal = journal
        self.settings = settings
        self.executor = ToolExecutor(broker, router, settings.risk, settings.watchlist)
        self.tools: list[dict[str, Any]] = list(TOOLS)
        if settings.enable_web_search:
            self.tools.append(WEB_SEARCH_TOOL)

    def _create(self, messages: list[dict[str, Any]]):
        return self.client.beta.messages.create(
            model=self.settings.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.settings.effort},
            # Caches the growing conversation, so each step in a cycle re-reads the prefix cheaply.
            cache_control={"type": "ephemeral"},
            # On a safety-classifier false positive, retry server-side on Anthropic's recommended model
            # instead of leaving the cycle without a decision.
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def run_cycle(self) -> CycleResult:
        clock = self.broker.clock()
        cycle_id = self.journal.start_cycle(self.settings.model, self.settings.mode, t=clock.now)
        self.router.cycle_id = cycle_id

        if self.router.risk.kill_switch_active():
            self.journal.end_cycle(cycle_id, "halted", 0, {}, "Kill switch active; cycle skipped.", t=clock.now)
            return CycleResult(cycle_id, "halted", 0, "Kill switch active; cycle skipped.")

        try:
            account = self.broker.get_account()
        except BrokerError as e:
            report = f"Broker unavailable, cycle skipped: {e}"
            self.journal.end_cycle(cycle_id, "broker_error", 0, {}, report, t=clock.now)
            return CycleResult(cycle_id, "broker_error", 0, report)
        self.journal.snapshot_equity(account.equity, account.cash, t=clock.now)
        previous = [(r["started_at"], r["report"]) for r in self.journal.recent_reports(2)]
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": kickoff_message(
                now=clock.now.isoformat(timespec="minutes"), mode=self.settings.mode,
                watchlist=self.settings.watchlist, limits_summary=limits_summary(self.settings.risk),
                previous_reports=previous, language=self.settings.report_language,
                web_search=self.settings.enable_web_search),
        }]

        usage: dict[str, int] = {}
        orders: list[dict[str, Any]] = []
        status, report, steps = "step_limit", "", 0
        response = None

        for steps in range(1, self.settings.max_agent_steps + 1):
            try:
                response = self._create(messages)
            except anthropic.RateLimitError as e:
                status, report = "api_error", f"Rate limited by the Claude API: {e.message}"
                break
            except anthropic.APIStatusError as e:
                status, report = "api_error", f"Claude API error {e.status_code}: {e.message}"
                break
            except anthropic.APIConnectionError as e:
                status, report = "api_error", f"Could not reach the Claude API: {e}"
                break

            self._add_usage(usage, response.usage)
            # Append the full content unchanged: thinking blocks must be passed back as-is.
            messages.append({"role": "assistant", "content": response.content})
            self._journal_thinking(cycle_id, response.content)

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    output, is_error = self.executor.execute(block.name, block.input)
                    self.journal.event(cycle_id, "tool_call",
                                       {"name": block.name, "input": block.input, "output": output,
                                        "is_error": is_error})
                    if block.name in {"place_order", "close_position"} and not is_error:
                        orders.append({"tool": block.name, "input": block.input, "result": output})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": output, "is_error": is_error})
                # All results for parallel calls go back in a single user message.
                messages.append({"role": "user", "content": results})
                continue
            if response.stop_reason == "pause_turn":
                continue  # a server-side tool (web search) paused; resending resumes the turn
            if response.stop_reason == "refusal":
                status = "refused"
                details = getattr(response, "stop_details", None)
                report = f"Model declined this cycle ({getattr(details, 'category', None)})."
                break
            if response.stop_reason == "max_tokens":
                status = "truncated"
                report = self._final_text(response.content) or "Response hit max_tokens."
                break
            status = "completed"
            report = self._final_text(response.content)
            break

        if status == "step_limit":
            report = f"Stopped after {self.settings.max_agent_steps} steps without a final report."

        self.journal.end_cycle(cycle_id, status, steps, usage, report, t=self.broker.clock().now)
        return CycleResult(cycle_id, status, steps, report, usage, orders)

    @staticmethod
    def _final_text(content: list[Any]) -> str:
        return "\n".join(b.text for b in content if b.type == "text").strip()

    def _journal_thinking(self, cycle_id: str, content: list[Any]) -> None:
        for b in content:
            if b.type == "thinking" and getattr(b, "thinking", ""):
                self.journal.event(cycle_id, "thinking", b.thinking)

    @staticmethod
    def _add_usage(total: dict[str, int], u: Any) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            total[key] = total.get(key, 0) + (getattr(u, key, 0) or 0)
