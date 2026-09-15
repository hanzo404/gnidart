"""تست‌های فاز ۴ — تشخیص رشتهٔ باخت و پست‌مورتم آماری."""
import unittest

import numpy as np
import pandas as pd

from bot.analysis.diagnostics import (diagnose_streak, p_loss_streak,
                                      p_wins_at_most)
from bot.analysis.postmortem import attach_regime, postmortem


def make_trades(n=40, wr=0.33, seed=5, tail_losses=3, start="2024-01-02 12:00"):
    """معاملات مصنوعی؛ tail_losses باخت آخر برای ساخت رشته."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    rows = []
    for i in range(n - tail_losses):
        win = rng.random() < wr
        rows.append({
            "entry_time": t0 + pd.Timedelta(minutes=45 * i),
            "exit_time": t0 + pd.Timedelta(minutes=45 * i + 30),
            "direction": 1, "pnl": 60.0 if win else -20.0,
            "r": 2.5 if win else -1.0, "reason": "target" if win else "stop",
            "spread_usd": 0.14,
        })
    for k in range(tail_losses):  # رشتهٔ باخت در انتها
        i = n - tail_losses + k
        rows.append({
            "entry_time": t0 + pd.Timedelta(minutes=45 * i),
            "exit_time": t0 + pd.Timedelta(minutes=45 * i + 30),
            "direction": 1, "pnl": -20.0, "r": -1.0, "reason": "stop",
            "spread_usd": 0.14,
        })
    return pd.DataFrame(rows)


def regime_frame(trades, value=0):
    times = pd.DatetimeIndex(trades["entry_time"])
    return pd.DataFrame({"time": times - pd.Timedelta(minutes=1),
                         "regime": np.full(len(trades), value, dtype=int)})


class TestStats(unittest.TestCase):
    def test_p_loss_streak(self):
        self.assertAlmostEqual(p_loss_streak(3, 0.33), 0.67 ** 3)

    def test_p_wins_at_most_exact(self):
        # P(X≤1 | n=4, p=0.5) = (1+4)/16 = 0.3125
        self.assertAlmostEqual(p_wins_at_most(1, 4, 0.5), 5 / 16)

    def test_p_wins_at_most_bounds(self):
        self.assertEqual(p_wins_at_most(4, 4, 0.5), 1.0)
        self.assertAlmostEqual(p_wins_at_most(0, 4, 0.5), 0.5 ** 4)


class TestDiagnose(unittest.TestCase):
    def test_insufficient_data(self):
        rep = diagnose_streak(make_trades(n=10), streak=3)
        self.assertEqual(rep.verdict, "INSUFFICIENT_DATA")

    def test_normal_fluctuation(self):
        tr = make_trades(n=60, wr=0.40, seed=11)   # سالم: ۴۰٪ برد
        rep = diagnose_streak(tr, streak=3, regime_df=regime_frame(tr),
                              baseline_wr=0.35)
        self.assertEqual(rep.verdict, "NORMAL_FLUCTUATION")

    def test_chaos_exposure(self):
        tr = make_trades(n=60, wr=0.35, seed=3)
        rg = regime_frame(tr, value=2)             # همهٔ ورودها در CHAOS
        rep = diagnose_streak(tr, streak=3, regime_df=rg, baseline_wr=0.35)
        self.assertEqual(rep.verdict, "CHAOS_EXPOSURE")

    def test_edge_decay(self):
        # ۳۰ معاملهٔ اخیر فقط ۲ برد → p بسیار کوچک → پوسیدگی لبه
        tr = make_trades(n=35, wr=0.5, seed=7)
        tr.loc[tr.index[-31:], "pnl"] = -20.0
        tr.loc[tr.index[-31:], "r"] = -1.0
        rep = diagnose_streak(tr, streak=3, regime_df=regime_frame(tr),
                              baseline_wr=0.35)
        self.assertEqual(rep.verdict, "EDGE_DECAY")

    def test_spread_problem(self):
        tr = make_trades(n=60, wr=0.35, seed=9)
        tr.loc[tr.index[-3:], "spread_usd"] = 0.60   # ۴ برابر میانگین
        rep = diagnose_streak(tr, streak=3, regime_df=regime_frame(tr),
                              baseline_wr=0.35)
        self.assertEqual(rep.verdict, "SPREAD_PROBLEM")

    def test_regime_shift(self):
        tr = make_trades(n=60, wr=0.35, seed=13)
        rg = regime_frame(tr, value=1)               # ورودها در TREND_UP
        rg.loc[rg.index[-1], "regime"] = 0           # الان RANGE شده
        rep = diagnose_streak(tr, streak=3, regime_df=rg, baseline_wr=0.35)
        self.assertEqual(rep.verdict, "REGIME_SHIFT")


class TestPostmortem(unittest.TestCase):
    def test_small_sample_continue(self):
        pm = postmortem(make_trades(n=10), baseline_wr=0.35)
        self.assertEqual(pm["recommendation"], "CONTINUE")

    def test_healthy_continue(self):
        tr = make_trades(n=100, wr=0.35, seed=21)
        pm = postmortem(tr, baseline_wr=0.35)
        self.assertEqual(pm["recommendation"], "CONTINUE")

    def test_broken_edge_halts(self):
        tr = make_trades(n=60, wr=0.33, seed=31)
        tr.loc[tr.index[-40:], "pnl"] = -20.0        # ۴۰ باخت متوالی اخیر
        tr.loc[tr.index[-40:], "r"] = -1.0
        pm = postmortem(tr, baseline_wr=0.35)
        self.assertEqual(pm["recommendation"], "HALT_REVIEW")

    def test_attach_regime_ffill(self):
        tr = make_trades(n=5, wr=0.5, seed=1)
        rg = pd.DataFrame({"time": [pd.Timestamp("2024-01-01")],
                           "regime": [1]})
        out = attach_regime(tr, rg)
        self.assertTrue((out["entry_regime"] == 1).all())  # ffill از قبل


if __name__ == "__main__":
    unittest.main()
