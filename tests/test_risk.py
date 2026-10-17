from agentrade.config import RiskLimits
from agentrade.models import Account, Order, OrderRequest, OrderStatus, OrderType, Position, Side
from agentrade.risk import RiskManager, projected_exposure


def account(equity=100_000.0, cash=100_000.0, day_start=None):
    return Account(equity=equity, cash=cash, buying_power=cash, day_start_equity=day_start or equity)


def check(order, price=100.0, acct=None, positions=None, open_orders=None, limits=None, orders_today=0,
          **rm_kwargs):
    acct = acct or account()
    positions = positions or {}
    prices = {s: p.market_price for s, p in positions.items()} | {order.symbol: price}
    exposure = projected_exposure(acct, positions, open_orders or [], prices)
    rm = RiskManager(limits or RiskLimits(), **rm_kwargs)
    return rm.check(order, price, acct, exposure, orders_today)


def buy(symbol="AAPL", qty=100, stop=95.0, **kw):
    return OrderRequest(symbol, Side.BUY, qty, stop_loss_price=stop, **kw)


def test_reasonable_buy_is_approved():
    d = check(buy(qty=100))  # $10k = 10% of equity
    assert d.approved, d.reasons


def test_position_size_limit_reports_how_much_fits():
    d = check(buy(qty=250))  # 25% > 20%
    assert not d.approved
    assert any("at most 200.00 more shares" in r for r in d.reasons)


def test_stop_loss_required_and_bounded():
    assert not check(buy(stop=None)).approved
    assert not check(buy(stop=101.0)).approved  # above entry
    d = check(buy(stop=80.0))  # 20% away, max 15%
    assert not d.approved and any("below entry" in r for r in d.reasons)


def test_short_selling_blocked_by_default():
    d = check(OrderRequest("AAPL", Side.SELL, 10))
    assert not d.approved and any("Short selling" in r for r in d.reasons)


def test_closing_a_position_is_always_allowed_even_after_daily_loss():
    pos = {"AAPL": Position("AAPL", 300, 120.0, 100.0)}
    acct = account(equity=90_000, cash=60_000, day_start=100_000)  # -10% day
    d = check(OrderRequest("AAPL", Side.SELL, 300), acct=acct, positions=pos)
    assert d.approved, d.reasons


def test_daily_loss_blocks_new_risk():
    acct = account(equity=96_000, cash=96_000, day_start=100_000)  # -4% > 3%
    d = check(buy(qty=10), acct=acct)
    assert not d.approved and any("Daily loss" in r for r in d.reasons)


def test_kill_switch_blocks_everything(tmp_path):
    kill = tmp_path / "KILL"
    kill.write_text("x")
    pos = {"AAPL": Position("AAPL", 10, 100.0, 100.0)}
    d = check(OrderRequest("AAPL", Side.SELL, 10), positions=pos, kill_switch_path=kill)
    assert not d.approved and "Kill switch" in d.reasons[0]


def test_watchlist_and_daily_order_count():
    assert not check(buy(symbol="GME"), allowed_symbols=["AAPL"]).approved
    assert not check(buy(), orders_today=20).approved


def test_open_orders_count_toward_exposure_and_cash():
    pending = Order("o1", OrderRequest("AAPL", Side.BUY, 150), OrderStatus.PENDING, submitted_at=None)
    # 150 pending + 100 new = 250 shares = 25% of equity
    d = check(buy(qty=100), open_orders=[pending])
    assert not d.approved and any("of equity" in r for r in d.reasons)


def test_insufficient_cash_and_order_notional_cap():
    d = check(buy(qty=150), acct=account(equity=100_000, cash=5_000))
    assert any("Insufficient cash" in r for r in d.reasons)
    limits = RiskLimits(max_order_notional=5_000)
    assert not check(buy(qty=100), limits=limits).approved


def test_limit_orders_are_checked_at_their_limit_price():
    d = check(buy(qty=210, stop=80.0, order_type=OrderType.LIMIT, limit_price=90.0), price=100.0)
    assert d.approved, d.reasons  # 210 * 90 = 18.9% of equity; stop 11% below the limit
