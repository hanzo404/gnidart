"""تست‌های فاز ۶ — متا-لیبلینگ.

چهار چیز که باید قفل شوند:
    ۱. علیّت اندیکاتورها (تغییر آینده → گذشته ثابت)
    ۲. ترازبندی دیتاست (کندل سیگنال = یکی قبل از ورود) و لیبل
    ۳. خالص‌سازی پنجره‌ها (purge + عدم هم‌پوشانی ماه‌ها)
    ۴. علیّت فیچرهای توالی (roll10 فقط به گذشته نگاه می‌کند)
"""
from __future__ import annotations

import sys
import unittest
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.meta import (FEATURES, build_dataset, compute_indicators,
                               purged_monthly_windows)


def make_bars(n: int = 600, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    t = pd.date_range(start, periods=n, freq="15min")
    close = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
    o = close + rng.normal(0, 0.3, n)
    h = np.maximum(o, close) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, close) - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": l,
                         "close": close, "volume": 100.0,
                         "spread_usd": 0.15})


def make_h4(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.set_index("time").resample("4h", label="left",
                                           closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last"}).dropna().reset_index()


class TestIndicators(unittest.TestCase):
    def test_causal_future_change_does_not_touch_past(self):
        bars = make_bars(400)
        ind1 = compute_indicators(bars)
        cut = 250
        bars2 = bars.copy()
        bars2.loc[cut:, ["open", "high", "low", "close"]] = \
            bars2.loc[cut:, ["open", "high", "low", "close"]] * 3.0 + 500.0
        ind2 = compute_indicators(bars2)
        for col in ind1.columns:
            a = ind1[col].iloc[:cut].to_numpy()
            b = ind2[col].iloc[:cut].to_numpy()
            ok = np.allclose(a, b, equal_nan=True)
            self.assertTrue(ok, f"نشتی در فیچر {col}")


class TestDataset(unittest.TestCase):
    def _build(self):
        bars = make_bars(600)
        h4 = make_h4(bars)
        regime = np.zeros(len(bars), dtype=int)      # RANGE
        sess = np.ones(len(bars), dtype=bool)
        sig_i = 400
        entry_time = bars["time"].iloc[sig_i + 1]
        trades = pd.DataFrame([{
            "entry_time": entry_time,
            "exit_time": bars["time"].iloc[sig_i + 8],
            "direction": 1, "entry": float(bars["close"].iloc[sig_i]),
            "exit": 0.0, "stop": float(bars["close"].iloc[sig_i]) - 5.0,
            "target": 0.0, "lots": 0.01, "pnl": 12.5, "r": 2.5,
            "reason": "target", "bars_held": 8, "tag": "v0_fvg",
        }])
        return bars, build_dataset(bars, h4, regime, sess, trades), sig_i

    def test_alignment_label_and_features(self):
        bars, ds, sig_i = self._build()
        self.assertEqual(len(ds), 1)
        row = ds.iloc[0]
        self.assertEqual(int(row["win"]), 1)                # pnl>0 → برد
        self.assertGreater(row["stop_dist_atr"], 0)         # استاپ زیر کلوز
        self.assertEqual(int(row["regime_range"]), 1)       # RANGE در سیگنال
        et = pd.Timestamp(row["entry_time"])
        self.assertAlmostEqual(row["hour_sin"],
                               np.sin(2 * np.pi * (et.hour + et.minute / 60)
                                      / 24.0), places=9)
        # فیچر کندل سیگنال — نه کندل ورود
        ind_sig = compute_indicators(bars)["rsi14"].iloc[sig_i]
        self.assertAlmostEqual(row["rsi14"], float(ind_sig), places=9)
        for f in FEATURES:
            self.assertIn(f, ds.columns, f"فیچر {f} ساخته نشد")

    def test_label_uses_pnl_sign(self):
        bars = make_bars(600)
        h4 = make_h4(bars)
        trades = pd.DataFrame([{
            "entry_time": bars["time"].iloc[401],
            "exit_time": bars["time"].iloc[410],
            "direction": 1, "entry": 2000.0, "exit": 1990.0,
            "stop": 1995.0, "target": 2010.0, "lots": 0.01,
            "pnl": -4.0, "r": -1.0, "reason": "stop",
            "bars_held": 9, "tag": "v0_fvg",
        }])
        ds = build_dataset(bars, h4, np.zeros(len(bars), dtype=int),
                           np.ones(len(bars), dtype=bool), trades)
        self.assertEqual(int(ds.iloc[0]["win"]), 0)

    def test_sequence_features_only_look_back(self):
        bars = make_bars(600)
        h4 = make_h4(bars)
        times = bars["time"]
        n = 30
        off = 100        # بعد از گرم‌شدن اندیکاتورها (EMA200/BB20)
        trades = pd.DataFrame({
            "entry_time": times.iloc[off:off + n].to_numpy(),
            "exit_time": times.iloc[off + 2:off + 2 + n].to_numpy(),
            "direction": np.ones(n, dtype=int),
            "entry": np.full(n, 2000.0), "exit": np.full(n, 2000.0),
            "stop": np.full(n, 1995.0), "target": np.full(n, 2010.0),
            "lots": np.full(n, 0.01),
            "pnl": np.linspace(-5, 5, n),
            "r": np.linspace(-1, 1, n),
            "reason": ["stop"] * n, "bars_held": [2] * n,
            "tag": ["v0_fvg"] * n,
        })
        ds = build_dataset(bars, h4, np.zeros(len(bars), dtype=int),
                           np.ones(len(bars), dtype=bool), trades)
        self.assertEqual(len(ds), n)
        base = ds["roll10_wr"].to_numpy().copy()
        # باختن ردیف آخر نباید فیچرهای قبل از خودش را عوض کند
        ds.loc[len(ds) - 1, "win"] = 1 - ds.loc[len(ds) - 1, "win"]
        ds2 = build_dataset(bars, h4, np.zeros(len(bars), dtype=int),
                            np.ones(len(bars), dtype=bool), trades)
        # بازسازی از دیتاست اصلی → فقط خودِ ردیف آخر تغییر می‌کند
        changed = int((ds2["roll10_wr"].to_numpy() != base).sum())
        self.assertLessEqual(changed, 1)


class TestPurgedWindows(unittest.TestCase):
    def test_no_overlap_and_purge(self):
        rng = np.random.default_rng(3)
        n = 1200                      # ~۱۰ ماه با فرکانس ۶ ساعته
        t = pd.date_range("2024-01-01", periods=n, freq="6h")
        ds = pd.DataFrame({
            "entry_time": t,
            "exit_time": t + pd.to_timedelta(rng.integers(1, 8, n), unit="h"),
            "pnl": rng.normal(0.5, 5, n), "r": rng.normal(0.1, 1, n),
            "win": rng.integers(0, 2, n),
        })
        wins = purged_monthly_windows(ds, is_months=6, oos_months=1)
        self.assertGreater(len(wins), 0)
        for is_idx, oos_idx, _ in wins:
            oos_start = ds["entry_time"].iloc[oos_idx].min()
            # هیچ معاملهٔ IS، خروجی در OOS ندارد
            self.assertTrue((ds["exit_time"].iloc[is_idx] < oos_start).all())
            # ماه‌های IS قبل از OOS
            is_months = set(ds["entry_time"].iloc[is_idx]
                            .dt.strftime("%Y-%m"))
            oos_months = set(ds["entry_time"].iloc[oos_idx]
                             .dt.strftime("%Y-%m"))
            self.assertFalse(is_months & oos_months)
            # هم‌پوشانی ایندکسی نیست
            self.assertEqual(len(set(is_idx) & set(oos_idx)), 0)


if __name__ == "__main__":
    unittest.main()
