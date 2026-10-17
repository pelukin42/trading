"""OrderRouter: the single path from any decision-maker (agent or strategy) to the broker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .broker.base import Broker, BrokerError
from .journal import Journal
from .models import Order, OrderRequest, RiskDecision
from .risk import RiskManager, projected_exposure


@dataclass
class RouteResult:
    request: OrderRequest
    decision: RiskDecision
    order: Order | None = None
    error: str | None = None

    @property
    def submitted(self) -> bool:
        return self.order is not None

    def to_dict(self) -> dict:
        r = self.request
        out = {
            "symbol": r.symbol, "side": r.side.value, "qty": r.qty, "order_type": r.order_type.value,
            "limit_price": r.limit_price, "stop_loss_price": r.stop_loss_price,
            "approved_by_risk": self.decision.approved,
        }
        if self.decision.reasons:
            out["risk_rejections"] = self.decision.reasons
        if self.order:
            out["broker_order_id"] = self.order.id
            out["status"] = self.order.status.value
        if self.error:
            out["broker_error"] = self.error
        return out


class OrderRouter:
    def __init__(self, broker: Broker, risk: RiskManager, journal: Journal):
        self.broker = broker
        self.risk = risk
        self.journal = journal
        self.cycle_id: str | None = None

    def _session_start(self, now: datetime) -> datetime:
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def route(self, request: OrderRequest) -> RouteResult:
        request.symbol = request.symbol.upper()
        now = self.broker.clock().now
        try:
            price = self.broker.get_latest_price(request.symbol)
            # Open orders first: one that fills between these calls is then counted twice (in open
            # orders and in positions/cash) rather than not at all, so a race can only over-restrict.
            open_orders = self.broker.get_open_orders()
            positions = self.broker.get_positions()
            account = self.broker.get_account()
        except BrokerError as e:
            decision = RiskDecision(False, [f"Could not load account state: {e}"])
            return self._record(RouteResult(request, decision), now)

        prices = {s: p.market_price for s, p in positions.items()}
        prices[request.symbol] = price
        for o in open_orders:
            if o.symbol not in prices and not o.request.limit_price:
                try:
                    prices[o.symbol] = self.broker.get_latest_price(o.symbol)
                except BrokerError:
                    pass
        exposure = projected_exposure(account, positions, open_orders, prices)
        orders_today = self.journal.approved_orders_since(self._session_start(now))
        decision = self.risk.check(request, price, account, exposure, orders_today)
        result = RouteResult(request, decision)
        if decision.approved:
            try:
                result.order = self.broker.submit_order(request)
            except BrokerError as e:
                result.error = str(e)
        return self._record(result, now)

    def _record(self, result: RouteResult, now: datetime) -> RouteResult:
        r = result.request
        self.journal.order(
            self.cycle_id, symbol=r.symbol, side=r.side.value, qty=r.qty, order_type=r.order_type.value,
            limit_price=r.limit_price, stop_loss_price=r.stop_loss_price,
            approved=result.decision.approved and result.error is None,
            risk_reasons=result.decision.reasons, broker_order_id=result.order.id if result.order else None,
            status=result.order.status.value if result.order else ("error" if result.error else "rejected"),
            rationale=r.rationale, t=now,
        )
        return result
