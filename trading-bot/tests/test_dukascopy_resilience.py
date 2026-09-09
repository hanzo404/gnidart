"""تست مقاومت دانلودر Dukascopy در برابر 503/خطای شبکه — بدون اینترنت.

درس عملی 2026-09-10: دانلود ۳-ساله با ۸ worker در روز ~۶۲م روی
BID_candles_min_1.bi5 با ConnectionError بعد از ۳ تلاش کرش کرد.
رفتار جدید مورد انتظار:
    - 503 تلاش مجدد با backoff می‌گیرد و معمولاً بعداً می‌گیرد
    - شکست پایدار = skip + ثبت در failed_days، بدون کرش کل ران
    - روز ناموفق در کش مارکر نمی‌گیرد (اجرای مجدد retry می‌کند)
    - 404 = روز بدون داده → مارکر خالی کش نوشته می‌شود
"""
import lzma
import struct
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from bot.data.dukascopy import DukascopyDownloader, day_url, parse_bi5_candles

_DAY = date(2023, 10, 12)   # همان روزِ خطای واقعی 503
_GOOD = date(2023, 10, 11)  # روز سالم کنارش


def make_bi5(rows):
    """ساخت فایل bi5 معتبر: [(sec, o, c, l, h, v), ...] — دقت: close قبل از low/high."""
    body = b"".join(struct.pack(">6i", *r) for r in rows)
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


class FakeResp:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """session.get با سناریوی اسکریپت‌شده: لیست status_code یا "raise"."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if step == "raise":
            import requests
            raise requests.ConnectionError("network down")
        return FakeResp(step)

    class headers:
        update = staticmethod(lambda *a, **k: None)


def make_dl(script, cache=None, **kw):
    dl = DukascopyDownloader("XAUUSD", cache_dir=cache,
                             retries=kw.pop("retries", 4),
                             backoff_max=0.0, retry_passes=0,
                             max_workers=1, **kw)
    dl.session = FakeSession(script)
    return dl


class TestRetryable(unittest.TestCase):
    def test_503_then_200_recovers(self):
        """503 اول → backoff → 200: داده برمی‌گردد، روز ناموفق نمی‌شود."""
        dl = make_dl([503, 503, 200])
        got = dl.fetch_day(_DAY)
        self.assertIsNotNone(got)
        self.assertNotIn(_DAY, dl.failed_days)

    def test_persistent_503_records_failure_no_raise(self):
        """503 همیشگی: نه raise، نه مارکر کش — فقط ثبت در failed_days."""
        with TemporaryDirectory() as tmp:
            dl = make_dl([503] * 10, cache=tmp)
            got = dl.fetch_day(_DAY)
            self.assertIsNone(got)
            self.assertIn(_DAY, dl.failed_days)
            # load_day نباید مارکر خالی بنویسد تا اجرای بعدی retry کند
            df = dl.load_day(_DAY)
            self.assertTrue(df.empty)
            self.assertEqual(list(Path(tmp).glob("*.csv.gz")), [],
                             "روز ناموفق نباید مارکر کش بگیرد")

    def test_404_is_normal_no_data(self):
        """404 = تعطیلی/آخر هفته → بدون خطا، مارکر کشِ خالی نوشته می‌شود."""
        with TemporaryDirectory() as tmp:
            dl = make_dl([404], cache=tmp)
            got = dl.fetch_day(_DAY)
            self.assertIsNone(got)
            self.assertNotIn(_DAY, dl.failed_days)
            dl.load_day(_DAY)
            self.assertEqual(len(list(Path(tmp).glob("*.csv.gz"))), 1,
                             "روز بدون داده باید مارکر خالی بگیرد")

    def test_network_exception_retries_then_fails(self):
        """استثنای شبکه مثل 503: retry، بعد skip بدون raise."""
        dl = make_dl(["raise"] * 10)
        got = dl.fetch_day(_DAY)
        self.assertIsNone(got)
        self.assertIn(_DAY, dl.failed_days)

    def test_bad_payload_marks_day_failed(self):
        """داده خراب (رکورد ناقص) → روز ناموفق، نه کرش."""
        dl = DukascopyDownloader("XAUUSD", retries=2, backoff_max=0.0)
        with self.assertRaises(lzma.LZMAError):
            parse_bi5_candles(b"\x00" * 10, _DAY)  # داده ناقص/خراب
        df = dl._parse_or_fail(b"\x00" * 10, _DAY)
        self.assertTrue(df.empty)
        self.assertIn(_DAY, dl.failed_days)

    def test_fetch_range_survives_bad_day(self):
        """fetch_range با یک روزِ همیشه-503 کرش نمی‌کند؛ روز سالم را برمی‌گرداند."""
        raw = make_bi5([(0, 1830000, 1831000, 1829000, 1832000, 500),
                        (60, 1831000, 1832000, 1830500, 1832500, 700)])
        with TemporaryDirectory() as tmp:
            dl = make_dl([503] * 20, cache=tmp)
            orig = dl.load_day

            def patched(day):
                if day == _GOOD:
                    return parse_bi5_candles(raw, day)
                return orig(day)  # _DAY → 503 همیشگی

            dl.load_day = patched
            df = dl.fetch_range(_GOOD, _DAY, progress_every=0)
            self.assertEqual(len(df), 2)
            self.assertEqual(df["time"].iloc[0], pd.Timestamp(_GOOD))
            self.assertIn(_DAY, dl.failed_days)

    def test_all_failed_no_frames_raises_connectionerror(self):
        """کل بازه ناموفق و هیچ فریمی نیست → ConnectionError (نه KeyError)."""
        with TemporaryDirectory() as tmp:
            dl = make_dl([503] * 20, cache=tmp)
            with self.assertRaises(ConnectionError):
                dl.fetch_range(_DAY, _DAY, progress_every=0)

    def test_retry_pass_recovers_failed_day(self):
        """پاس تلاش مجدد انتهایی، روز جا‌مانده را نجات می‌دهد و کش می‌کند."""
        raw = make_bi5([(0, 1830000, 1831000, 1829000, 1832000, 500)])
        with TemporaryDirectory() as tmp:
            dl = DukascopyDownloader("XAUUSD", cache_dir=tmp, retries=1,
                                     backoff_max=0.0, retry_passes=1,
                                     max_workers=1)
            state = {"calls": 0}

            def get(url, timeout=None):
                state["calls"] += 1
                # فاز اصلی: 503 → پاس retry: 200
                return FakeResp(503) if state["calls"] == 1 else FakeResp(200, raw)

            dl.session = FakeSession.__new__(FakeSession)
            dl.session.get = get
            with mock.patch("time.sleep"):  # backoff واقعی را fast کن
                df = dl.fetch_range(_DAY, _DAY, progress_every=0)
            self.assertEqual(len(df), 1)
            self.assertNotIn(_DAY, dl.failed_days)
            self.assertTrue(any(Path(tmp).glob("*.csv.gz")),
                            "روز نجات‌یافته باید کش شود")

    def test_url_month_is_zero_based(self):
        self.assertIn("/2023/09/12/", day_url("XAUUSD", date(2023, 10, 12)))


if __name__ == "__main__":
    unittest.main()
