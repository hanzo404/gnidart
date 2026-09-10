"""تست‌های آماده‌سازی داده: فیلتر پدینگ duka، تشخیص آفست سرور، M15."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from bot.data.prepare import (attach_spread_profile, infer_offset_minutes,
                              load_duka, load_mt5, shift_mt5_to_utc, to_m15)


def m1_df(start, n, base=2000.0):
    t = pd.date_range(start, periods=n, freq="1min")
    px = base + np.sin(np.arange(n) / 50.0) * 5
    return pd.DataFrame({"time": t, "open": px, "high": px + 1, "low": px - 1,
                         "close": px + 0.5, "volume": 10.0})


class TestLoaders(unittest.TestCase):
    def test_load_duka_drops_padding(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.csv.gz"
            real = pd.concat([m1_df("2024-01-01", 10), m1_df("2024-01-02", 10)],
                             ignore_index=True)
            # ۳ کندل پدینگ: حجم صفر و OHLC کاملاً تخت
            pad = pd.DataFrame({"time": pd.date_range("2024-01-03", periods=3, freq="1min"),
                                "open": 123.0, "high": 123.0, "low": 123.0,
                                "close": 123.0, "volume": 0.0})
            pd.concat([real, pad], ignore_index=True).to_csv(
                p, index=False, compression="gzip")
            out = load_duka(p)
            self.assertEqual(len(out), 20)
            self.assertTrue((out["volume"] > 0).all())

    def test_load_mt5_spread_conversion(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.csv.gz"
            df = m1_df("2024-01-01", 5)
            df = df.rename(columns={"volume": "tick_volume"})
            df["spread"] = 14  # پوینت → $0.14
            df.to_csv(p, index=False, compression="gzip")
            out = load_mt5(p)
            self.assertAlmostEqual(out["spread_usd"].iloc[0], 0.14, places=6)
            self.assertIn("volume", out.columns)


class TestOffset(unittest.TestCase):
    def test_infer_offset_finds_180(self):
        duka = m1_df("2024-03-01", 5000)                    # UTC
        mt5 = m1_df("2024-03-01", 5000)                     # همان قیمت‌ها
        mt5 = mt5.rename(columns={"volume": "tick_volume"})
        mt5["spread"] = 14
        mt5["time"] = mt5["time"] + pd.Timedelta(minutes=180)  # سرور = UTC+3
        offs = infer_offset_minutes(mt5, duka)
        self.assertEqual(offs.get("2024-03"), 180)

    def test_shift_to_utc(self):
        mt5 = m1_df("2024-03-01", 100)
        mt5["time"] = mt5["time"] + pd.Timedelta(minutes=180)
        out = shift_mt5_to_utc(mt5, {"2024-03": 180})
        self.assertEqual(out["time"].iloc[0], pd.Timestamp("2024-03-01 00:00"))


class TestM15(unittest.TestCase):
    def test_to_m15_aggregation_and_spread_first(self):
        rows = []
        for i in range(30):
            rows.append((i, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 5.0, 0.10 + 0.01 * i))
        df = pd.DataFrame(rows, columns=["min", "open", "high", "low", "close", "volume", "spread_usd"])
        df["time"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df.pop("min"), unit="m")
        out = to_m15(df)
        self.assertEqual(len(out), 2)
        b0 = out.iloc[0]
        self.assertAlmostEqual(b0["open"], 100.0)
        self.assertAlmostEqual(b0["high"], 115.0)   # max دقیقه‌های ۰..۱۴ (۱۰۱+i، i=14)
        self.assertAlmostEqual(b0["low"], 99.0)
        self.assertAlmostEqual(b0["close"], 114.5)
        self.assertAlmostEqual(b0["volume"], 75.0)
        self.assertAlmostEqual(b0["spread_usd"], 0.10)  # اولین دقیقه

    def test_attach_spread_profile(self):
        bars = pd.DataFrame({"time": [pd.Timestamp("2024-01-01 03:00")],
                             "open": [1.0], "high": [1.0], "low": [1.0],
                             "close": [1.0], "volume": [1.0]})
        prof = pd.Series([0.1] * 24)
        prof.iloc[3] = 0.5
        out = attach_spread_profile(bars, prof)
        self.assertAlmostEqual(out["spread_usd"].iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
