"""تست پایپ‌لاین داده: پارسر bi5 (ساختگی)، ریسمپل، لودر CSV محلی."""
import sys
import pathlib
import unittest
from datetime import date, datetime

import lzma
import struct

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bot.data.dukascopy import parse_bi5_candles, day_url
from bot.data.resample import resample, build_timeframes
from bot.data.local import load_ohlc_csv


def make_bi5(records):
    """ساخت فایل bi5 ساختگی: رکورد ۲۴ بایتی [sec, open, close, low, high, vol]."""
    buf = b"".join(struct.pack(">6i", *r) for r in records)
    return lzma.compress(buf, format=lzma.FORMAT_ALONE)


class TestBi5Parser(unittest.TestCase):
    DAY = date(2026, 8, 6)

    def test_parse_three_candles(self):
        # دقت ترتیب Dukascopy: open, CLOSE, low, high
        raw = make_bi5([
            (0,     4361000, 4362000, 4360500, 4362500, 1500),   # 00:00
            (60,    4362000, 4361500, 4361000, 4362600, 900),    # 00:01
            (120,   4361500, 4363000, 4361400, 4363100, 2200),   # 00:02
        ])
        df = parse_bi5_candles(raw, self.DAY, price_scale=1000.0, volume_scale=1000.0)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["time"].iloc[0], pd.Timestamp("2026-08-06 00:00:00"))
        self.assertEqual(df["time"].iloc[2], pd.Timestamp("2026-08-06 00:02:00"))
        self.assertAlmostEqual(df["open"].iloc[0], 4361.0)
        self.assertAlmostEqual(df["close"].iloc[0], 4362.0)   # ستون دوم = close!
        self.assertAlmostEqual(df["low"].iloc[0], 4360.5)
        self.assertAlmostEqual(df["high"].iloc[0], 4362.5)
        self.assertAlmostEqual(df["volume"].iloc[0], 1.5)
        self.assertAlmostEqual(df["high"].iloc[2], 4363.1)

    def test_empty_and_none(self):
        self.assertTrue(parse_bi5_candles(b"", self.DAY).empty)
        self.assertTrue(parse_bi5_candles(lzma.compress(b"", format=lzma.FORMAT_ALONE), self.DAY).empty)

    def test_url_month_is_zero_based(self):
        self.assertIn("/XAUUSD/2026/00/05/", day_url("XAUUSD", date(2026, 1, 5)))


class TestResample(unittest.TestCase):
    def _m1(self, n=60):
        base = pd.Timestamp("2026-08-06 00:00:00")
        rows = []
        for i in range(n):
            o = 100.0 + i
            rows.append({
                "time": base + pd.Timedelta(minutes=i),
                "open": o, "high": o + 1.0, "low": o - 1.0,
                "close": o + 0.5, "volume": 10.0,
            })
        return pd.DataFrame(rows)

    def test_m1_to_m5(self):
        df = resample(self._m1(60), "M5")
        self.assertEqual(len(df), 12)
        first = df.iloc[0]
        self.assertEqual(first["time"], pd.Timestamp("2026-08-06 00:00:00"))
        self.assertAlmostEqual(first["open"], 100.0)          # اولین open
        self.assertAlmostEqual(first["close"], 104.5)         # آخرین closeِ ۵ کندل
        self.assertAlmostEqual(first["high"], 105.0)          # max (100+4)+1
        self.assertAlmostEqual(first["low"], 99.0)            # min (100)-1
        self.assertAlmostEqual(first["volume"], 50.0)         # جمع ۵ کندل

    def test_partial_bin_dropped_when_no_data(self):
        df = resample(self._m1(62), "M5")
        # ۶۲ کندل = ۱۲ بازه کامل + ۲ کندل؛ بازه ناقصِ آخر هم ساخته می‌شود (open دارد)
        # — چون داده‌اش موجود است. قرارداد ما: در لایو، کندل ناقص را قبل ریسمپل حذف کن.
        self.assertEqual(len(df), 13)

    def test_build_timeframes(self):
        out = build_timeframes(self._m1(120), ("M5", "M15", "H1"))
        self.assertEqual(len(out["M5"]), 24)
        self.assertEqual(len(out["M15"]), 8)
        self.assertEqual(len(out["H1"]), 2)


class TestLocalLoader(unittest.TestCase):
    def test_mt5_schema(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "mt5.csv")
            lines = ["time,open,high,low,close,tick_volume,spread"]
            for i in range(10):
                t = (pd.Timestamp("2026-08-06 00:00:00") + pd.Timedelta(minutes=i))
                lines.append(f"{t.strftime('%Y.%m.%d %H:%M')},4361.0,4362.0,4360.0,4361.5,100,25")
            # یک ردیف تکراری + یک تخلف OHLC
            lines.append(f"{(pd.Timestamp('2026-08-06 00:00:00')).strftime('%Y.%m.%d %H:%M')},4361.0,4362.0,4360.0,4361.5,100,25")
            lines.append("2026.08.06 00:12,4363.0,4362.0,4360.0,4361.5,100,25")  # open خارج بازه [low,high]!
            with open(p, "w") as f:
                f.write("\n".join(lines))
            df, rep = load_ohlc_csv(p)
            self.assertEqual(len(df), 11)                       # ۱۲ ردیف - ۱ تکراری
            self.assertEqual(rep["duplicates_dropped"], 1)
            self.assertEqual(rep["ohlc_violations"], 1)
            self.assertEqual(df["time"].iloc[0], pd.Timestamp("2026-08-06 00:00:00"))
            self.assertIn("volume", df.columns)                 # tick_volume → volume
            self.assertAlmostEqual(df["volume"].iloc[0], 100.0)

    def test_missing_column_raises(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad.csv")
            with open(p, "w") as f:
                f.write("time,open,close\n2026-08-06,1,2\n")
            with self.assertRaises(ValueError):
                load_ohlc_csv(p)


if __name__ == "__main__":
    unittest.main()
