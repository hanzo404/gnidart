"""تست ریسک‌فری خودکار (be_at_frac) — ایده‌ی وام‌دارِ EA خارجی.

صحنه: خرید در 2000، استاپ 1990، TP 2025. قیمت تا 60٪ مسیر TP بالا می‌رود
(close ≥ 2015) → استاپ باید به نقطه‌ی ورود منتقل شود → وقتی برمی‌گردد،
خروج در نقطه‌ی ورود (زیانِ حدوداً صفر) نه استاپِ کامل −1R.
بدون BE همان صحنه باید −1R کامل ببندد (رفتار پیش‌فرض دست‌نخورده).
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester, Order


def make_bars() -> pd.DataFrame:
    """۳۰ کندل ۱۵ دقیقه‌ای: گرم‌شدن → رالی تا 2016 → بازگشت تا 1986.

    open هر کندل = close قبلی (استاندارد)؛ سیگنال در i=5 → ورود در open
    کندل 6 = 2000. ریسک = 2000−1990 = $10، TP = 2025، نصف مسیر = 2012.5
    (close کندل 9 = 2014 تریگر می‌شود).
    """
    t = pd.date_range("2024-01-02 12:00", periods=30, freq="15min")
    closes = [2000.0] * 6                       # گرم‌شدن (سیگنال در i=5)
    closes += [2005, 2009, 2012, 2014, 2016]    # رالی
    closes += [2012, 2008, 2002, 1998, 1996]    # بازگشت آرام
    closes += [1994, 1992, 1986]                # شکست (بدون BE: استاپ کامل)
    closes += [1986] * (30 - len(closes))
    closes = [float(c) for c in closes[:30]]
    o = [closes[0]] + closes[:-1]               # open = close قبلی
    h = [max(oc, c) + 1.0 for oc, c in zip(o, closes)]
    l = [min(oc, c) - 1.0 for oc, c in zip(o, closes)]
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": l,
                         "close": closes, "volume": 1.0})


class SignalOnce:
    """در i=5 سیگنال خرید می‌دهد (استاپ 1990، RR=2.5)."""
    def __init__(self):
        self._n = 0

    def prepare(self, bars):
        self._n = len(bars)

    def on_bar(self, i):
        return Order(+1, 1990.0, 2.5, "test_be") if i == 5 else None


class TestBreakeven(unittest.TestCase):
    def _run(self, be: float | None):
        bars = make_bars()
        cfg = BacktestConfig(start_equity=3000.0, fixed_lots=0.01,
                             slippage_usd=0.0, be_at_frac=be)
        res = Backtester(bars, SignalOnce(), cfg).run()
        self.assertEqual(len(res.trades), 1)
        return res.trades.iloc[0]

    def test_be_exit_near_entry_instead_of_full_loss(self):
        tr = self._run(0.5)
        self.assertEqual(tr["reason"], "stop")
        # خروج در نقطه‌ی ورود (± لغزشِ صفر) → زیانِ تقریباً صفر، نه −1R
        self.assertGreater(tr["pnl"], -2.0)
        self.assertLess(abs(tr["pnl"]), 2.0)

    def test_without_be_full_stop_loss(self):
        tr = self._run(None)
        self.assertEqual(tr["reason"], "stop")
        # استاپ کامل: خروج ~1990 → زیان کامل (~$10 با 0.01 لات)
        self.assertLess(tr["pnl"], -8.0)

    def test_default_config_unchanged(self):
        self.assertIsNone(BacktestConfig().be_at_frac)


if __name__ == "__main__":
    unittest.main()
