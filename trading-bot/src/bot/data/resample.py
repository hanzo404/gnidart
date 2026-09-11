"""ریسمپل M1 → تایم‌فریم‌های بالاتر (M5/M15/H1/...).

قواعد تجمیع OHLCV استاندارد: open=اولین، high=بیشینه، low=کمینه،
close=آخرین، volume=جمع. بازه‌ها [left, right) و برچسب = ابتدای بازه.

قرارداد ضد-نشتی: خروجیِ ریسمپل روی داده تاریخی کامل است؛ در اجرای زنده،
همیشه «آخرین کندلِ در حال تشکیل» را قبل از تصمیم حذف کن
(همان کاری که MT5DataProvider با closed_only=True می‌کند).
"""
from __future__ import annotations

import pandas as pd

TIMEFRAME_RULES = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """df باید ستون‌های time/open/high/low/close/volume داشته باشد."""
    rule = TIMEFRAME_RULES.get(timeframe.upper())
    if rule is None:
        raise ValueError(f"تایم‌فریم پشتیبانی نمی‌شود: {timeframe} (مجاز: {list(TIMEFRAME_RULES)})")
    if df.empty:
        return df.copy()
    out = (
        df.set_index("time")
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )
    return out


def build_timeframes(df_m1: pd.DataFrame, timeframes=("M5", "M15", "H1")) -> dict:
    """M1 خام → دیکشنری {timeframe: df}. ورودی باید M1 باشد."""
    return {tf: resample(df_m1, tf) for tf in timeframes}
