"""تست‌های فاز ۳ — اتصال بریکر و سایزینگ %-ریسک به موتور بک‌تست."""
import unittest

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestConfig, Backtester, Order
from bot.backtest.risk import BreakerPolicy
from bot.risk.circuit_breaker import CircuitBreaker


def make_bars(rows):
    df = pd.DataFrame(rows, columns=["min", "open", "high", "low", "close"])
    df["time"] = pd.Timestamp("2024-01-02 00:00") + pd.to_timedelta(df.pop("min"), unit="m")
    df["volume"] = 100.0
    df["spread_usd"] = 0.10
    return df


class AlwaysBuy:
    """همیشه خرید با استاپ ۵ دلاری زیر قیمت جاری."""

    def prepare(self, bars):
        self.closes = bars["close"].to_numpy()

    def on_bar(self, i):
        return Order(+1, stop=float(self.closes[i] - 5.0), rr=2.0)


class TestBreakerPolicy(unittest.TestCase):
    def _policy(self):
        return BreakerPolicy(CircuitBreaker(pause_hours=24.0), grade="B")

    def test_derate_and_restore_ladder(self):
        p = self._policy()
        p.on_trade_closed(-1.0, "2024-01-02 10:00", 3000)
        p.on_trade_closed(-1.0, "2024-01-02 11:00", 2990)
        self.assertAlmostEqual(p.size_multiplier(), 0.5)   # ۲ باخت → نصف
        p.on_trade_closed(+2.0, "2024-01-02 12:00", 3010)
        self.assertAlmostEqual(p.size_multiplier(), 1.0)   # برد → یک پله برگشت

    def test_deep_derate_blocks_b_grade(self):
        p = self._policy()
        for k in range(3):
            p.on_trade_closed(-1.0, f"2024-01-02 1{k}:00", 3000)
        self.assertAlmostEqual(p.size_multiplier(), 0.25)
        self.assertFalse(p.allow_entry("2024-01-02 15:00"))  # B-grade بلاک

    def test_pause_blocks_then_expires(self):
        p = BreakerPolicy(CircuitBreaker(pause_hours=24.0), grade="A")
        for k in range(4):
            p.on_trade_closed(-1.0, "2024-01-02 10:00", 3000)
        self.assertFalse(p.allow_entry(pd.Timestamp("2024-01-02 20:00")))  # داخل ۲۴ ساعت
        self.assertTrue(p.allow_entry(pd.Timestamp("2024-01-04 10:01")))   # بعدش باز

    def test_monthly_halt_and_auto_ack(self):
        p = self._policy()
        p.on_bar_close(pd.Timestamp("2024-03-01 00:00"), 3000.0)   # شروع ماه
        p.on_bar_close(pd.Timestamp("2024-03-20 00:00"), 2780.0)   # −۷.۳٪ → halt
        self.assertTrue(p.breaker.halted)
        self.assertFalse(p.allow_entry(pd.Timestamp("2024-03-20 01:00")))
        p.on_bar_close(pd.Timestamp("2024-04-01 00:00"), 2780.0)   # مرز ماه → auto-ack
        self.assertFalse(p.breaker.halted)
        self.assertTrue(p.allow_entry(pd.Timestamp("2024-04-01 01:00")))
        self.assertIn("2024-03", p.halted_months)


class TestEngineWithPolicy(unittest.TestCase):
    def _falling_bars(self, n=140):
        """بازار نزولی پیوسته — خریدِ همیشه = رشته باخت.""" ""
        rows = []
        px = 2000.0
        for i in range(n):
            o = px
            px -= 2.0
            rows.append((i, o, o + 0.5, px - 0.5, px))
        return make_bars(rows)

    def test_sizing_derates_after_losses(self):
        bars = self._falling_bars()
        pol = BreakerPolicy(CircuitBreaker())
        cfg = BacktestConfig(start_equity=30_000.0, risk_pct=0.01,
                             max_positions=1, cooldown_bars=0,
                             slippage_usd=0.05, min_lots=0.01, lot_step=0.01)
        res = Backtester(bars, AlwaysBuy(), cfg, risk_policy=pol).run()
        self.assertTrue(len(res.trades) >= 4)
        lots = res.trades["lots"].to_numpy()
        # نردبان: ۱.۰ → بعد از ۲ باخت ×۰.۵ → بعد از ۳ باخت ×۰.۲۵
        self.assertGreater(lots[0], lots[2])   # قبل از derate > بعد از آن
        self.assertLessEqual(lots[2], lots[1])  # کاهش‌ها ناپیوسته نیستند
        self.assertGreater(pol.counters.get("derate", 0), 0)
        self.assertGreater(pol.counters.get("deep_derate", 0), 0)
        self.assertGreater(pol.counters.get("pause", 0), 0)   # ۴ باخت → مکث

    def test_pause_blocks_entries(self):
        bars = self._falling_bars(160)
        pol = BreakerPolicy(CircuitBreaker())
        cfg = BacktestConfig(start_equity=30_000.0, risk_pct=0.01,
                             max_positions=1, cooldown_bars=0,
                             slippage_usd=0.05)
        res = Backtester(bars, AlwaysBuy(), cfg, risk_policy=pol).run()
        self.assertGreater(pol.counters.get("pause", 0), 0)
        self.assertGreater(pol.counters.get("entry_blocks", 0), 0)
        # کندل‌ها ۱-دقیقه‌ای‌اند → مکث ۲۴h (=۱۴۴۰ کندل) بقیهٔ تست را می‌پوشاند
        self.assertLessEqual(len(res.trades), 6)

    def test_min_lot_clamp_counted(self):
        """حساب کوچک: کاهش بریکر از کف لات رد می‌شود و شمرده می‌شود."""
        bars = self._falling_bars()
        pol = BreakerPolicy(CircuitBreaker())
        cfg = BacktestConfig(start_equity=3000.0, risk_pct=0.005,
                             max_positions=1, cooldown_bars=0,
                             slippage_usd=0.05)
        res = Backtester(bars, AlwaysBuy(), cfg, risk_policy=pol).run()
        # اولین معامله ۰.۰۲ (ریسک $15 ÷ فاصله ~7.15)؛ بعد از deep-derate (×۰.۲۵)
        # زیر کف می‌رود → clamp به ۰.۰۱ و شمرده می‌شود
        self.assertAlmostEqual(res.trades["lots"].iloc[0], 0.02, places=6)
        self.assertEqual(res.trades["lots"].min(), 0.01)
        self.assertGreaterEqual(res.metrics.get("min_lot_clamps", 0), 1)

    def test_no_policy_backward_compatible(self):
        bars = self._falling_bars(40)
        cfg = BacktestConfig(start_equity=3000.0, fixed_lots=0.01,
                             max_positions=1, cooldown_bars=0, slippage_usd=0.05)
        res = Backtester(bars, AlwaysBuy(), cfg).run()
        self.assertTrue(len(res.trades) > 0)  # بدون policy مثل قبل کار می‌کند


if __name__ == "__main__":
    unittest.main()
