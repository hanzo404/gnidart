"""تست‌های موتور رژیم — علیّت، برچسب‌ها، سشن، و گیت استراتژی."""
import unittest

import numpy as np
import pandas as pd

from bot.backtest.engine import Backtester, BacktestConfig, Order
from bot.backtest.gate import GatedStrategy
from bot.regime.engine import (CHAOS, RANGE, RegimeConfig, RegimeEngine,
                               TREND_DOWN, TREND_UP, session_open)


def bars_from(closes, start="2024-01-01"):
    closes = np.asarray(closes, dtype=float)
    t = pd.date_range(start, periods=len(closes), freq="15min")
    return pd.DataFrame({"time": t, "open": closes, "high": closes * 1.001,
                         "low": closes * 0.999, "close": closes,
                         "volume": 100.0})


class Scripted:
    def __init__(self, sig):
        self.sig = sig

    def prepare(self, bars):
        pass

    def on_bar(self, i):
        return self.sig


class ScriptedAt(Scripted):
    def __init__(self, signals):
        self.signals = signals

    def prepare(self, bars):
        pass

    def on_bar(self, i):
        return self.signals.get(i)


class TestRegimeLabels(unittest.TestCase):
    def test_uptrend_detected(self):
        rng = np.random.default_rng(7)
        closes = 2000 + np.cumsum(rng.normal(2.0, 0.8, 400))  # روند صعودی تمیز
        eng = RegimeEngine()
        out = eng.compute(bars_from(closes))
        tail = out["regime"].iloc[-100:]
        self.assertGreater((tail == TREND_UP).mean(), 0.8)

    def test_downtrend_detected(self):
        rng = np.random.default_rng(8)
        closes = 2000 - np.cumsum(rng.normal(2.0, 0.8, 400))
        eng = RegimeEngine()
        out = eng.compute(bars_from(closes))
        tail = out["regime"].iloc[-100:]
        self.assertGreater((tail == TREND_DOWN).mean(), 0.8)

    def test_oscillation_is_range(self):
        rng = np.random.default_rng(12)
        t = np.arange(600)
        closes = 2000 + 6 * np.sin(t / 60.0) + rng.normal(0, 1.0, 600)  # جعبهٔ رنج واقعی: دامنه هم‌مقیاس نویز
        eng = RegimeEngine()
        out = eng.compute(bars_from(closes))
        tail = out["regime"].iloc[-200:]
        self.assertGreater((tail == RANGE).mean(), 0.5)

    def test_vol_explosion_is_chaos(self):
        rng = np.random.default_rng(9)
        base = 2000 + rng.normal(0, 0.3, 600)          # آرام
        burst = 2000 + rng.normal(0, 25.0, 60)         # انفجار نوسان شدید
        closes = np.concatenate([base, burst])
        eng = RegimeEngine()
        out = eng.compute(bars_from(closes))
        last20 = out["regime"].iloc[-20:]
        self.assertGreater((last20 == CHAOS).mean(), 0.5)

    def test_causality_prefix_invariance(self):
        """رژیم کندل‌های اول نباید با اضافه‌شدن دادهٔ آینده عوض شود."""
        rng = np.random.default_rng(11)
        closes = 2000 + np.cumsum(rng.normal(1.0, 1.5, 800))
        full = bars_from(closes)
        eng = RegimeEngine()
        k = 500
        pre = eng.compute(full.iloc[:k])
        ful = eng.compute(full)
        np.testing.assert_array_equal(pre["regime"].to_numpy(),
                                      ful["regime"].to_numpy()[:k])


class TestSession(unittest.TestCase):
    def test_session_hours_utc(self):
        t = pd.date_range("2024-01-02", periods=96, freq="15min")  # یک شبانه‌روز
        m = session_open(t)
        self.assertFalse(m[0])         # 00:00 بسته
        self.assertTrue(m[48])         # 12:00 باز
        self.assertFalse(m[80])        # 20:00 بسته
        self.assertFalse(m[84])        # 21:00 (رول‌اور) بسته


class TestGate(unittest.TestCase):
    def _bars(self):
        t = pd.date_range("2024-01-02 12:00", periods=20, freq="15min")
        c = 2000 + np.arange(20) * 0.5
        return pd.DataFrame({"time": t, "open": c, "high": c + 1, "low": c - 1,
                             "close": c, "volume": 100.0,
                             "spread_usd": 0.1})

    def test_gate_blocks_against_regime_and_off_session(self):
        bars = self._bars()
        regime = np.full(20, TREND_UP)
        sess = np.ones(20, dtype=bool)
        inner = Scripted(Order(+1, stop=1990.0))       # خرید — هم‌جهت
        gated = GatedStrategy(inner, regime, sess)
        gated.prepare(bars)
        self.assertIsNotNone(gated.on_bar(5))
        inner_sell = Scripted(Order(-1, stop=2010.0))  # فروش — خلاف رژیم
        gated2 = GatedStrategy(inner_sell, regime, sess)
        gated2.prepare(bars)
        self.assertIsNone(gated2.on_bar(5))
        sess_closed = np.zeros(20, dtype=bool)          # خارج از سشن
        gated3 = GatedStrategy(inner, regime, sess_closed)
        gated3.prepare(bars)
        self.assertIsNone(gated3.on_bar(5))
        gated4 = GatedStrategy(inner, np.full(20, CHAOS), sess)  # آشفتگی
        gated4.prepare(bars)
        self.assertIsNone(gated4.on_bar(5))

    def test_gate_still_exits_positions(self):
        """گیت فقط ورود را می‌بندد؛ استاپِ پوزیشن باز در سشن بسته هم فعال است."""
        bars = self._bars()
        c = 2000 + np.arange(20) * 0.5
        c[15:] = 1990.0                          # سقوط بعد از بسته‌شدن سشن
        bars["open"] = c
        bars["high"] = c + 1
        bars["low"] = c - 1
        bars["close"] = c
        regime = np.full(20, TREND_UP)
        sess = np.ones(20, dtype=bool)
        sess[8:] = False                         # از کندل ۸ سشن بسته
        inner = ScriptedAt({1: Order(+1, stop=1995.0, rr=2.0)})
        gated = GatedStrategy(inner, regime, sess)
        res = Backtester(bars, gated, BacktestConfig(
            slippage_usd=0.05, max_positions=1)).run()
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(res.trades.iloc[0]["reason"], "gap_stop")
        self.assertLess(res.trades.iloc[0]["pnl"], 0)


if __name__ == "__main__":
    unittest.main()
