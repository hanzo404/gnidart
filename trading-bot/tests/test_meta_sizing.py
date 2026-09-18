"""تست‌های فاز ۶-ب — سایزینگ احتمالاتی.

سه چیز را قفل می‌کنیم:
    ۱. توابع سایزینگ: کلامپ [0.5, 1.5]، هیچ‌گاه صفر، لبه‌های باندها
    ۲. agg_weighted: ریاضی PF/PnL وزنی و DD
    ۳. run_meta_sizing_wfo روی دیتاست مصنوعی: ستون‌های m کامل و محدود
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.meta import FEATURES, MetaWFOConfig
from bot.analysis.meta_sizing import (agg_weighted, run_meta_sizing_wfo,
                                      sizing_kelly, sizing_linear,
                                      sizing_stepped)


class TestSizingFunctions(unittest.TestCase):
    def test_stepped_bands(self):
        p = np.array([0.10, 0.34, 0.35, 0.44, 0.45, 0.90])
        np.testing.assert_allclose(
            sizing_stepped(p), [0.5, 0.5, 1.0, 1.0, 1.5, 1.5])

    def test_linear_clamped_never_zero(self):
        p = np.array([0.0, 0.05, 0.34, 0.6, 1.0])
        m = sizing_linear(p, 0.34)
        np.testing.assert_allclose(m, [0.5, 0.5, 1.0, 1.5, 1.5])
        self.assertTrue((m >= 0.5).all() and (m <= 1.5).all())

    def test_kelly_normalized_and_clamped(self):
        # p = p_base → m = 1.0؛ کلی مثبت در نرخ‌های برد معمول
        m = sizing_kelly(np.array([0.34, 0.10, 0.60]), 0.34)
        self.assertAlmostEqual(m[0], 1.0, places=6)
        self.assertEqual(m[1], 0.5)
        self.assertEqual(m[2], 1.5)
        # نرخ برد پایهٔ خیلی پایین → fallback به linear (بدون تقسیم بر منفی)
        m2 = sizing_kelly(np.array([0.2, 0.3]), 0.20)
        np.testing.assert_allclose(m2, sizing_linear(np.array([0.2, 0.3]), 0.20))


class TestAggWeighted(unittest.TestCase):
    def test_math(self):
        # ۲ برد (+100، +250) و ۲ باخت (−100، −100) با وزن‌های متفاوت
        frame = pd.DataFrame({
            "pnl": [100.0, 250.0, -100.0, -100.0],
            "r": [2.5, 2.5, -1.0, -1.0],
            "m_x": [1.5, 0.5, 1.0, 0.5],
        })
        a = agg_weighted(frame, "m_x", n_windows=10)
        # سود وزنی: 150+125=275 | ضرر وزنی: 100+50=150 → PF=1.8333
        self.assertAlmostEqual(a["pf"], 275 / 150, places=4)
        self.assertAlmostEqual(a["pnl"], 275 - 150, places=4)
        # انتظار سرمایه‌وزنی: (3.75+1.25-1-0.5)/(1.5+0.5+1+0.5)
        self.assertAlmostEqual(a["exp_r"], 3.5 / 3.5, places=4)
        self.assertAlmostEqual(a["mean_m"], 3.5 / 4, places=4)
        self.assertAlmostEqual(a["per_month"], 0.4, places=4)

    def test_empty(self):
        a = agg_weighted(pd.DataFrame(), "m_x", 5)
        self.assertEqual(a["n"], 0)


def _synthetic_ds(n_months: int = 16, per_month: int = 6,
                  seed: int = 7) -> pd.DataFrame:
    """دیتاست مصنوعی با همهٔ ستون‌های لازم (FEATURES + win/r/pnl/زمان‌ها)."""
    rng = np.random.default_rng(seed)
    rows = []
    for mth in range(n_months):
        base = pd.Timestamp("2024-01-01") + pd.DateOffset(months=mth)
        for i in range(per_month):
            t = base + pd.Timedelta(hours=12 * i)
            win = bool(rng.random() < 0.35)
            r = 2.5 if win and rng.random() < 0.7 else (-1.0 if not win
                                                        else 0.5)
            row = {f: float(rng.normal()) for f in FEATURES}
            row.update({"entry_time": t, "exit_time": t,
                        "win": int(win), "r": r, "pnl": r * 10.0})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


class TestWFO(unittest.TestCase):
    def test_synthetic_run_bounds(self):
        ds = _synthetic_ds()
        res = run_meta_sizing_wfo(
            ds, MetaWFOConfig(thresholds=(0.5,), min_kept_is=30))
        self.assertGreaterEqual(len(res.per_window), 1)
        self.assertIn(res.per_window["model"].iloc[0],
                      {"logit", "gbm", "base_rate"})
        for col in ("m_stepped", "m_linear", "m_kelly", "m_oracle"):
            self.assertIn(col, res.oos.columns)
            m = res.oos[col].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(m).all())
            self.assertTrue(((m >= 0.5) & (m <= 1.5)).all(),
                            f"{col} خارج از کلامپ")
            # هیچ معامله‌ای حذف نمی‌شود
            self.assertEqual(len(res.oos), res.per_window["oos_n"].sum())
        # احتمال‌ها معتبر
        self.assertTrue(((res.oos["p"] >= 0) & (res.oos["p"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
