"""تست‌های آداپتور MT5 — v0.5.8 (ممیزی ۴، P0-2).

پکیج MetaTrader5 فقط روی ویندوز نصب می‌شود؛ اینجا ماژولِ جعلی را داخل
ماژول آداپتور تزریق می‌کنیم (monkeypatch) و فقط «منطق» را می‌سنجیم:
  1) گیت stops_level قبل از ارسال — استاپ نزدیک → رد شفاف، نه 10016
  2) پیش‌پرواز order_check — ردِ سرور → سفارش واقعی ارسال نمی‌شود
  3) مسیر سالم — order_check ✓ → order_send ✓ → Fill برمی‌گردد
  4) retry روی retcode گذرا همچنان کار می‌کند
"""
import unittest
from types import SimpleNamespace

import bot.execution.mt5_adapter as mod
from bot.execution.base import Order


class _SymbolInfo(SimpleNamespace):
    pass


class FakeMT5:
    """mt5 جعلی — همهٔ چیزهایی که آداپتور لمس می‌کند."""

    ORDER_FILLING_IOC = 2
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    ACCOUNT_TRADE_MODE_DEMO = 0
    POSITION_TYPE_BUY = 0

    def __init__(self, stops_level=0, point=0.01, bid=2000.0, spread=0.14):
        self.stops_level = stops_level
        self.point = point
        self.bid = bid
        self.ask = bid + spread
        self.order_check_calls = []
        self.order_send_calls = []
        self.check_retcode = 0          # 0 = تایید
        self.send_retcode = self.TRADE_RETCODE_DONE

    def symbol_info(self, symbol):
        return _SymbolInfo(trade_stops_level=self.stops_level,
                           point=self.point, filling_mode=2)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=self.ask, bid=self.bid)

    def account_info(self):
        return SimpleNamespace(trade_mode=self.ACCOUNT_TRADE_MODE_DEMO)

    def order_check(self, req):
        self.order_check_calls.append(req)
        return SimpleNamespace(retcode=self.check_retcode, comment="")

    def order_send(self, req):
        self.order_send_calls.append(req)
        return SimpleNamespace(retcode=self.send_retcode, order=777,
                               price=req.get("price", 0.0))


def _adapter(fake):
    """ساخت آداپتور بدون __init__ (چون _HAS_MT5 در لینوکس False است)."""
    a = object.__new__(mod.MT5ExecutionAdapter)
    a.symbol = "XAUUSD"
    a.magic = 954001
    a.deviation = 10
    a.require_demo = False
    mod.mt5 = fake          # تزریق
    mod._HAS_MT5 = True
    return a


def _order(stop=1990.0, target=2025.0):
    return Order(symbol="XAUUSD", direction=1, units=0.01,
                 price=2000.0, stop=stop, target=target)


class TestStopsLevelGate(unittest.TestCase):
    def setUp(self):
        self._saved = (mod.mt5, mod._HAS_MT5)

    def tearDown(self):
        mod.mt5, mod._HAS_MT5 = self._saved

    def test_close_stop_rejected_before_any_send(self):
        fake = FakeMT5(stops_level=50, point=0.01)   # حداقل 0.50$
        a = _adapter(fake)
        # اسپرد 0.14 → قیمت ورود ~2000.14؛ استاپ در 2000.0 → فاصله 0.14 < 0.50
        with self.assertRaises(RuntimeError) as ctx:
            a.place_order(_order(stop=2000.00, target=2025.0))
        self.assertIn("stops_level", str(ctx.exception))
        self.assertEqual(fake.order_send_calls, [])   # هیچ ارسالی نکرد
        self.assertEqual(fake.order_check_calls, [])  # حتی چک هم لازم نبود

    def test_zero_stops_level_passes(self):
        fake = FakeMT5(stops_level=0)
        a = _adapter(fake)
        fill = a.place_order(_order())
        self.assertEqual(len(fake.order_send_calls), 1)
        self.assertEqual(fill.order_id, "777")

    def test_far_stop_passes(self):
        fake = FakeMT5(stops_level=50, point=0.01)   # حداقل 0.50$
        a = _adapter(fake)
        fill = a.place_order(_order(stop=1990.0, target=2025.0))
        self.assertEqual(fill.order_id, "777")       # فاصله ~10$ > 0.50$


class TestOrderCheckPreflight(unittest.TestCase):
    def setUp(self):
        self._saved = (mod.mt5, mod._HAS_MT5)

    def tearDown(self):
        mod.mt5, mod._HAS_MT5 = self._saved

    def test_rejected_preflight_blocks_send(self):
        fake = FakeMT5()
        fake.check_retcode = 10019   # NOT_ENOUGH_MONEY
        a = _adapter(fake)
        with self.assertRaises(RuntimeError) as ctx:
            a.place_order(_order())
        self.assertIn("order_check", str(ctx.exception))
        self.assertEqual(fake.order_send_calls, [])  # ارسال واقعی نشد

    def test_ok_preflight_then_send(self):
        fake = FakeMT5()
        a = _adapter(fake)
        fill = a.place_order(_order())
        self.assertEqual(len(fake.order_check_calls), 1)
        self.assertEqual(len(fake.order_send_calls), 1)
        self.assertEqual(fill.order_id, "777")


class TestTransientRetryStillWorks(unittest.TestCase):
    def setUp(self):
        self._saved = (mod.mt5, mod._HAS_MT5)

    def tearDown(self):
        mod.mt5, mod._HAS_MT5 = self._saved

    def test_requote_then_success(self):
        fake = FakeMT5()
        fake.send_retcode = fake.TRADE_RETCODE_REQUOTE
        a = _adapter(fake)

        state = {"n": 0}

        def _send(req):
            state["n"] += 1
            fake.order_send_calls.append(req)
            rc = (fake.TRADE_RETCODE_REQUOTE if state["n"] == 1
                  else fake.TRADE_RETCODE_DONE)
            return SimpleNamespace(retcode=rc, order=888,
                                   price=req.get("price", 0.0))

        fake.order_send = _send
        fill = a.place_order(_order())
        self.assertEqual(state["n"], 2)      # دوبار تلاش شد
        self.assertEqual(fill.order_id, "888")


if __name__ == "__main__":
    unittest.main()
