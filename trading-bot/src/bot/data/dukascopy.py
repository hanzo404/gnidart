"""داده تاریخی Dukascopy (فرمت bi5) — عمق بالا و رایگان برای بک‌تست.

⚠️ این ماژول روی ماشین خودت اجرا می‌شود (سندباکس توسعه به datafeed دسترسی ندارد).

فرمت bi5 (هر فایل = یک روز، BID، مبنای UTC):
    - LZMA-alone فشرده
    - رکوردهای ۲۴ بایتی big-endian، ۶ عدد int32:
      [ثانیه از نیمه‌شب، open، close، low، high، volume]
      (دقت: close قبل از low/high می‌آید — peculiarity Dukascopy)
    - قیمت‌ها برای XAUUSD با 10^3 مقیاس می‌شوند (۳ رقم اعشار)
    - حجم‌ها با 10^3 مقیاس می‌شوند (واحد: لات)

سیاست مقاومت در برابر خطا (درس خطای 503 در دانلود ۳-ساله):
    - 404  = روز بدون داده (تعطیلی/آخر هفته) → عادی؛ مارکر کش خالی نوشته می‌شود
    - 429/502/503/504 = rate-limit یا سرویس موقتاً در دسترس نیست
        → backoff نمایی (۲، ۴، ۸، ۱۶، ۳۲، حداکثر ۴۵ ثانیه) تا ۶ بار
    - روزِ همچنان ناموفق → در failed_days ثبت، skip می‌شود و در پایان
      پاس‌های تلاش مجدد (ترتیبی و آرام) دوباره امتحان می‌شود
    - کرش کل ران ممنوع؛ کشِ روزانه هم یعنی اجرای مجدد از همان‌جا ادامه می‌یابد
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import requests

_REC = struct.Struct(">6i")  # sec_offset, open, close, low, high, volume
_HEADERS = {"User-Agent": "Mozilla/5.0 (gnidart-trading-bot research)"}
_BASE = "https://datafeed.dukascopy.com/datafeed"
# خطاهایی که تلاش مجدد منطقی است (rate-limit / در دسترس نبودن موقت سرویس)
_RETRYABLE_STATUS = {429, 502, 503, 504}

# مقیاس قیمت بر حسب سیمبل (رقم اعشار Dukascopy)
PRICE_SCALES = {
    "XAUUSD": 1000.0,
    "XAGUSD": 1000.0,
    "EURUSD": 100000.0, "GBPUSD": 100000.0, "XAUJPY": 1000.0,
}
VOLUME_SCALE = 1000.0

_COLS = ["time", "open", "high", "low", "close", "volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def day_url(instrument: str, day: date) -> str:
    """ماه در URL صفر-مبنا است (ژانویه = 00) — نوعی تله معروف."""
    return f"{_BASE}/{instrument}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"


def parse_bi5_candles(raw: bytes, day: date,
                      price_scale: float = 1000.0,
                      volume_scale: float = VOLUME_SCALE) -> pd.DataFrame:
    """خامِ فایل روزانه → DataFrame کندل‌های M1 (UTC، naive)."""
    if not raw:
        return _empty()
    data = lzma.decompress(raw)
    if not data:
        return _empty()
    if len(data) % _REC.size:
        raise ValueError(f"طول داده bi5 با رکورد ۲۴ بایتی سازگار نیست: {len(data)}")
    rows = [_REC.unpack_from(data, off) for off in range(0, len(data), _REC.size)]
    df = pd.DataFrame(rows, columns=["off", "open", "close", "low", "high", "volume"])
    base = pd.Timestamp(day)
    df["time"] = base + pd.to_timedelta(df.pop("off"), unit="s")
    for c in ("open", "close", "low", "high"):
        df[c] = df[c] / price_scale
    df["volume"] = df["volume"] / volume_scale
    return df[_COLS]


class DukascopyDownloader:
    """دانلود بازه تاریخچه با کش روزانه (csv.gz) و دانلود موازیِ مؤدبانه.

    failed_days: روز‌هایی که بعد از همه retryها هم نگرفته شدند (شبکه/503).
    این روزها عمداً در کش مارکر نمی‌گیرند تا اجرای بعدی دوباره تلاش کند.
    """

    def __init__(self, instrument: str = "XAUUSD", cache_dir: Optional[str] = None,
                 timeout: int = 25, retries: int = 6, max_workers: int = 3,
                 retry_passes: int = 3, backoff_max: float = 45.0) -> None:
        self.instrument = instrument.upper()
        self.price_scale = PRICE_SCALES.get(self.instrument, 100000.0)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.max_workers = max_workers
        self.retry_passes = retry_passes
        self.backoff_max = backoff_max
        self.failed_days: Set[date] = set()
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    # ------------------------------------------------------------------ #
    def fetch_day(self, day: date) -> Optional[bytes]:
        """خام یک روز.

        None + day ∉ failed_days → 404 (روز بدون داده؛ عادی)
        None + day ∈ failed_days  → شکست شبکه/سرویس پس از همه تلاش‌ها
        """
        url = day_url(self.instrument, day)
        last_err: object = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    return None
                if r.status_code not in _RETRYABLE_STATUS:
                    r.raise_for_status()
                    return r.content
                last_err = f"HTTP {r.status_code} (rate-limit/موقتی)"
            except requests.RequestException as e:
                last_err = e
            time.sleep(min(self.backoff_max, 2.0 * (2 ** attempt)))
        self.failed_days.add(day)
        return None

    def _cache_path(self, day: date) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self.instrument}_{day.isoformat()}.csv.gz"

    def _parse_or_fail(self, raw: bytes, day: date) -> pd.DataFrame:
        """پارس با بیمه: داده خراب = روزِ ناموفق، نه کرشِ کل ران."""
        try:
            return parse_bi5_candles(raw, day, self.price_scale)
        except Exception:
            self.failed_days.add(day)
            return _empty()

    def load_day(self, day: date) -> pd.DataFrame:
        """یک روز با کش. روزِ ناموفق مارکرِ خالی نمی‌گیرد (تا retry شود)."""
        cp = self._cache_path(day)
        if cp is not None and cp.exists():
            df = pd.read_csv(cp, parse_dates=["time"])
            return df if not df.empty else _empty()
        raw = self.fetch_day(day)
        if raw is None:
            if day in self.failed_days:
                return _empty()  # شبکه — مارکر ننویس
            df = _empty()        # 404 — واقعاً بدون داده، مارکر بنویس
        else:
            df = self._parse_or_fail(raw, day)
            if day in self.failed_days:
                return df
        if cp is not None:
            df.to_csv(cp, index=False, compression="gzip")
        return df

    def fetch_range(self, start: date, end: date, progress_every: int = 30) -> pd.DataFrame:
        """بازه [start, end] → یک DataFrame مرتب و بدون تکرار.

        هرگز به‌خاطر چند روز خطادار کرش نمی‌کند؛ در پایان پاس‌های
        تلاش مجدد دارد و اگر باز ماند، فقط هشدار می‌دهد.
        """
        if end < start:
            raise ValueError("end قبل از start است")
        days: List[date] = []
        d = start
        while d <= end:
            days.append(d)  # شنبه هم می‌فرستیم — 404 خودش هندل می‌شود
            d += timedelta(days=1)

        from concurrent.futures import ThreadPoolExecutor

        frames: List[pd.DataFrame] = []
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for df in ex.map(self.load_day, days):
                if not df.empty:
                    frames.append(df)
                done += 1
                if progress_every and done % progress_every == 0:
                    msg = f"  ... {done}/{len(days)} روز پردازش شد"
                    if self.failed_days:
                        msg += f" (ناموفق فعلی: {len(self.failed_days)} — بعداً retry می‌شود)"
                    print(msg)

        # پاس‌های تلاش مجدد برای روزهای خطادار — ترتیبی و آرام‌تر
        for pass_no in range(1, self.retry_passes + 1):
            if not self.failed_days:
                break
            retry_days = sorted(self.failed_days)
            self.failed_days = set()
            print(f"  ↻ تلاش مجدد {pass_no}/{self.retry_passes} برای {len(retry_days)} روز...")
            time.sleep(10.0 * pass_no)
            for day in retry_days:
                raw = self.fetch_day(day)
                if raw is None:
                    if day not in self.failed_days:  # این بار 404 بود → بدون داده
                        cp = self._cache_path(day)
                        if cp is not None:
                            _empty().to_csv(cp, index=False, compression="gzip")
                    continue
                df = self._parse_or_fail(raw, day)
                if day in self.failed_days:
                    continue
                frames.append(df)
                cp = self._cache_path(day)
                if cp is not None:
                    df.to_csv(cp, index=False, compression="gzip")

        if self.failed_days:
            sample = ", ".join(str(x) for x in sorted(self.failed_days)[:5])
            print(f"  ⚠️ {len(self.failed_days)} روز هنوز در دسترس نبود (مثلاً {sample}).")
            print("     دوباره همان دستور را اجرا کن — کش ادامه می‌دهد و فقط همین‌ها را می‌گیرد.")

        if not frames:
            if self.failed_days:
                raise ConnectionError(
                    f"هیچ داده‌ای دریافت نشد؛ {len(self.failed_days)} روز ناموفق "
                    f"(شبکه/rate-limit). بعداً دوباره تلاش کن."
                )
            return _empty()

        full = pd.concat(frames, ignore_index=True)
        return (
            full.drop_duplicates(subset="time", keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
