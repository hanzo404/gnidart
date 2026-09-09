"""داده تاریخی Dukascopy (فرمت bi5) — عمق بالا و رایگان برای بک‌تست.

⚠️ نکته اجرا: این ماژول روی هر ماشینی با اینترنت آزاد اجرا می‌شود
(روی ویندوز تو با `scripts/fetch_dukascopy.py`). محیط توسعه سندباکس فقط
به دامنه‌های allowlist دسترسی دارد؛ برای همین دانلود در سندباکس ممکن نیست.

فرمت bi5 (هر فایل = یک روز، BID، مبنای UTC):
    - LZMA-alone فشرده
    - رکوردهای ۲۴ بایتی big-endian، ۶ عدد int32:
      [ثانیه از نیمه‌شب، open، close، low، high، volume]
      (دقت: close قبل از low/high می‌آید — частности Dukascopy)
    - قیمت‌ها برای XAUUSD با 10^3 مقیاس می‌شوند (۳ رقم اعشار)
    - حجم‌ها با 10^3 مقیاس می‌شوند (واحد: لات)
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

_REC = struct.Struct(">6i")  # sec_offset, open, close, low, high, volume
_HEADERS = {"User-Agent": "Mozilla/5.0 (gnidart-trading-bot research)"}
_BASE = "https://datafeed.dukascopy.com/datafeed"

# مقیاس قیمت بر حسب سیمبل (رقم اعشار Dukascopy)
PRICE_SCALES = {
    "XAUUSD": 1000.0,
    "XAGUSD": 1000.0,
    # جفت‌های ۵ رقمی فارکس:
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
    """دانلود بازه تاریخچه با کش روزانه (csv.gz) و دانلود موازی."""

    def __init__(self, instrument: str = "XAUUSD", cache_dir: Optional[str] = None,
                 timeout: int = 25, retries: int = 3, max_workers: int = 8) -> None:
        self.instrument = instrument.upper()
        self.price_scale = PRICE_SCALES.get(self.instrument, 100000.0)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    # ------------------------------------------------------------------ #
    def fetch_day(self, day: date) -> Optional[bytes]:
        """خام یک روز؛ None یعنی بدون داده (تعطیلی/آخر هفته)."""
        url = day_url(self.instrument, day)
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException as e:  # شبکه موقت
                last_err = e
                time.sleep(1.0 + attempt)  # backoff کوچک
        raise ConnectionError(f"دانلود ناموفق پس از {self.retries} تلاش: {url} ({last_err})")

    def _cache_path(self, day: date) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self.instrument}_{day.isoformat()}.csv.gz"

    def load_day(self, day: date) -> pd.DataFrame:
        """یک روز با کش: فایل خالی=بدون داده (تا دوباره ۴۰۴ نگیریم)."""
        cp = self._cache_path(day)
        if cp is not None and cp.exists():
            df = pd.read_csv(cp, parse_dates=["time"])
            return df if not df.empty else _empty()
        raw = self.fetch_day(day)
        df = _empty() if raw is None else parse_bi5_candles(raw, day, self.price_scale)
        if cp is not None:
            df.to_csv(cp, index=False, compression="gzip")
        return df

    def fetch_range(self, start: date, end: date, progress_every: int = 30) -> pd.DataFrame:
        """بازه [start, end] → یک DataFrame مرتب و بدون تکرار."""
        if end < start:
            raise ValueError("end قبل از start است")
        days: List[date] = []
        d = start
        while d <= end:
            days.append(d)  # شنبه هم می‌فرستیم — 404 خودش هندل می‌شود
            d += timedelta(days=1)

        from concurrent.futures import ThreadPoolExecutor

        out: List[pd.DataFrame] = []
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for df in ex.map(self.load_day, days):
                if not df.empty:
                    out.append(df)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  ... {done}/{len(days)} روز پردازش شد")
        if not out:
            return _empty()
        full = pd.concat(out, ignore_index=True)
        full = full.drop_duplicates(subset="time", keep="last").sort_values("time")
        return full.reset_index(drop=True)
