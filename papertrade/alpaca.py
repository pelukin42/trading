"""Cliente mínimo de Alpaca restringido a la cuenta PAPER (dinero ficticio)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import quote

import requests

from papertrade.engine import Order

PAPER_URL = "https://paper-api.alpaca.markets"


class BrokerError(RuntimeError):
    pass


@dataclass
class AccountSnapshot:
    cash: float
    equity: float
    positions: dict[str, float]
    prices: dict[str, float]


def _format_qty(qty: float) -> str:
    truncated = math.floor(qty * 1e6 + 1e-6) / 1e6
    return f"{truncated:.6f}".rstrip("0").rstrip(".")


class AlpacaPaperBroker:
    def __init__(self, key: str, secret: str, base_url: str = PAPER_URL, session: requests.Session | None = None):
        if base_url.rstrip("/") != PAPER_URL:
            raise BrokerError(f"Solo se permite la cuenta paper de Alpaca ({PAPER_URL}). Operar con dinero real está bloqueado.")
        self.base_url = PAPER_URL
        self.session = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}/v2{path}"
        resp = self.session.request(method, url, headers=self.headers, timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise BrokerError(f"Alpaca {method} {path} → HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else None

    def account(self) -> AccountSnapshot:
        acct = self._request("GET", "/account")
        positions = self._request("GET", "/positions") or []
        return AccountSnapshot(
            cash=float(acct["cash"]),
            equity=float(acct["equity"]),
            positions={p["symbol"]: float(p["qty"]) for p in positions},
            prices={p["symbol"]: float(p["current_price"]) for p in positions},
        )

    def cancel_open_orders(self) -> None:
        self._request("DELETE", "/orders")

    def submit(self, order: Order) -> dict:
        if order.close_position:
            return self._request("DELETE", f"/positions/{quote(order.symbol, safe='')}") or {}
        body = {
            "symbol": order.symbol,
            "qty": _format_qty(order.qty),
            "side": order.side,
            "type": "market",
            "time_in_force": "day",
        }
        return self._request("POST", "/orders", json=body)
