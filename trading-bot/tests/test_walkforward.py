"""تست‌های هارنس walk-forward — ضد-نشتی پنجره‌ها، انتخاب، مونت‌کارلو."""
import unittest

import numpy as np
import pandas as pd

from bot.analysis.walkforward import (WFOConfig, WalkForwardOptimizer,
                                      bootstrap_maxdd, monthly_windows,
                                      select_best)


class TestWindows(unittest.TestCase):
    def test_no_overlap_and_order(self):
        """هیچ پنجرهٔ IS نباید با OOS هم‌پوشانی یا بعد از آن باشد."""
        keys = np.array([f"2024-{m:02d}" for m in range(1, 13)]
                        + [f"2025-{m:02d}" for m in range(1, 13)])
        wins = monthly_windows(keys, is_months=6, oos_months=1)
        # اولین OOS = 2024-07 (بعد از ۶ ماه IS) → تا 2025-12 = ۱۸ پنجره
        self.assertEqual(len(wins), 18)
        self.assertEqual(keys[wins[0][1]][0], "2024-07")
        for is_mask, oos_mask in wins:
            self.assertFalse((is_mask & oos_mask).any())
            is_months_set = set(keys[is_mask])
            oos_months_set = set(keys[oos_mask])
            self.assertTrue(all(i < o for i in is_months_set
                                for o in oos_months_set))

    def test_oos_months_grouping(self):
        keys = np.array([f"2024-{m:02d}" for m in range(1, 13)])
        wins = monthly_windows(keys, is_months=4, oos_months=2)
        self.assertEqual(len(wins), 4)  # 2+2+2+2
        for _, oos_mask in wins:
            self.assertEqual(oos_mask.sum(), 2)

    def test_insufficient_history(self):
        keys = np.array(["2024-01", "2024-02"])
        self.assertEqual(monthly_windows(keys, 12, 1), [])


class TestSelectBest(unittest.TestCase):
    def test_min_trades_guard(self):
        scores = [((21, 1.7, 2.0), 500.0, 10)]      # سود بالا ولی ۱۰ معامله
        best = select_best(scores, defaults=(23, 1.8, 2.5), min_trades=30)
        self.assertEqual(best, (23, 1.8, 2.5))      # → پیش‌فرض

    def test_best_pnl_wins(self):
        scores = [((21, 1.7, 2.0), 100.0, 50),
                  ((23, 1.8, 2.5), 300.0, 40),
                  ((25, 1.9, 3.0), 200.0, 60)]
        best = select_best(scores, (23, 1.8, 2.5), 30)
        self.assertEqual(best, (23, 1.8, 2.5))

    def test_negative_pnl_still_selectable_if_only_option(self):
        """اگر همه ضررده باشند، کمترین ضرر انتخاب می‌شود (نه پیش‌فرض کورکورانه)."""
        scores = [((21, 1.7, 2.0), -100.0, 40),
                  ((25, 1.9, 3.0), -300.0, 40)]
        best = select_best(scores, (23, 1.8, 2.5), 30)
        self.assertEqual(best, (21, 1.7, 2.0))


class TestBootstrap(unittest.TestCase):
    def test_quantiles_ordered_and_deterministic(self):
        rng = np.random.default_rng(1)
        pnls = rng.normal(5, 20, 120)
        a = bootstrap_maxdd(pnls, paths=500, seed=7)
        b = bootstrap_maxdd(pnls, paths=500, seed=7)
        for k in ("maxdd_p5", "maxdd_p50", "maxdd_p95"):
            self.assertEqual(a[k], b[k])            # seed → تکرارپذیر
        self.assertLessEqual(a["maxdd_p5"], a["maxdd_p50"])
        self.assertLessEqual(a["maxdd_p50"], a["maxdd_p95"])
        self.assertLess(a["maxdd_p5"], 0)           # افت منفی است

    def test_all_wins_never_dd(self):
        pnls = np.full(100, 10.0)
        mc = bootstrap_maxdd(pnls, paths=100, seed=1)
        self.assertEqual(mc["maxdd_p95"], 0.0)
        self.assertEqual(mc["p_dd_below_15"], 0.0)

    def test_empty(self):
        self.assertEqual(bootstrap_maxdd(np.array([])), {})


class TestOptimizerOnSynthetic(unittest.TestCase):
    """دو ماه دادهٔ مصنوعی کوچک → هارنس کامل باید بدون خطا و بدون نشتی اجرا شود."""

    def _bars(self):
        rng = np.random.default_rng(3)
        n = 6000   # ≈ ۶۲ روز → دو ماه تقویمی (ژانویه IS، فوریه OOS)
        t = pd.date_range("2024-01-01", periods=n, freq="15min")
        # روند ملایم + نویز → چند معامله رخ می‌دهد
        c = 2000 + np.cumsum(rng.normal(0.2, 3.0, n))
        return pd.DataFrame({"time": t, "open": c, "high": c + 2,
                             "low": c - 2, "close": c, "volume": 100.0,
                             "spread_usd": 0.1})

    def test_run_smoke_and_causality(self):
        bars = self._bars()
        h4 = pd.DataFrame({"time": [bars["time"].iloc[0]],
                           "open": [2000.0], "high": [2001.0],
                           "low": [1999.0], "close": [2000.5]})
        cfg = WFOConfig(is_months=1, oos_months=1, min_is_trades=0,
                        mc_paths=50)
        wfo = WalkForwardOptimizer(bars, h4, cfg)
        res = wfo.run()
        # ۶۲ روز ≈ سه ماه: فوریه و مارس OOS می‌شوند
        self.assertEqual(len(res.per_window), 2)
        # OOS فقط از ماه بعد از IS است
        keys = bars["time"].dt.strftime("%Y-%m").to_numpy()
        self.assertEqual(res.per_window.iloc[0]["oos_month"], "2024-02")
        self.assertTrue((keys == "2024-02").sum() > 0)


if __name__ == "__main__":
    unittest.main()
