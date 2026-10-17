from datetime import datetime, timedelta, timezone

import pytest

from agentrade.models import Bar


def make_bars(closes, start=datetime(2024, 1, 2, 21, tzinfo=timezone.utc), spread=0.01, opens=None):
    bars = []
    for i, c in enumerate(closes):
        o = opens[i] if opens else c
        bars.append(Bar(start + timedelta(days=i), o, max(o, c) * (1 + spread), min(o, c) * (1 - spread), c, 1_000_000))
    return bars


@pytest.fixture
def bars_factory():
    return make_bars
