"""بارگذاری داده و کالیبراسیون سشن.

قرارداد ستون‌ها: time (UTC naive), open, high, low, close, [volume|tick_volume]
دو فرمت ورودی پشتیبانی می‌شود (هر دو در ریشهٔ ریپو موجودند):
    - Dukascopy:  us30_m1_duka.csv.gz  → ستون volume
    - MT5 export: ustec_m1.csv.gz      → ستون tick_volume + spread
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class Spec:
    """مشخصات نماد: هزینه و اندازهٔ قرارداد برای هر نماد جدا تنظیم می‌شود."""

    symbol: str
    file: str
    point: float          # کوچک‌ترین تغییر قیمت (مثال: 0.01 برای US30)
    spread_pts: float     # اسپرد معمول در نقطه، روی M1
    slippage_pts: float   # اسلیپیج معمول در نقطه، هر بار
    contract: float = 1.0  # ارزش هر ۱.۰ حرکت قیمت به ازای هر واحد حجم


# مقادیر واقع‌گرایانه برای CFDهای شاخصی با حساب ECN.
# (برای طلا باید با اسپرد بروکر خودتان جایگزین شود — تنها گزینهٔ ناایمن)
SPECS = {
    "US30": Spec("US30", "us30_m1_duka.csv.gz", point=0.01, spread_pts=3.0, slippage_pts=1.0),
    "NAS100": Spec("NAS100", "usatechidxusd_m1_duka.csv.gz", point=0.01, spread_pts=2.5, slippage_pts=1.0),
}


def load_m1(spec: Spec, path: str | None = None) -> pd.DataFrame:
    """خواندن M1 با نرمال‌سازی نام ستون‌ها."""
    p = path or os.path.join(REPO, spec.file)
    df = pd.read_csv(p, compression="gzip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    if "spread" not in df.columns:
        df["spread"] = spec.spread_pts * spec.point
    keep = ["time", "open", "high", "low", "close", "volume", "spread"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "spread"):
        df[c] = df[c].astype(np.float64)
    return df


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """M1 → M{n}. برچسب کندل = شروع آن (UTC)."""
    if minutes == 1:
        return df.copy()
    rule = f"{minutes}min"
    out = (
        df.set_index("time")
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    )
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def valid_bars(df: pd.DataFrame, spec: Spec, min_range_pts: float = 0.5) -> pd.Series:
    """ماسک «کندل بی‌اطلاعات».

    حدود ۱۰٪ کندل‌های M1 این CFDها (سشن آسیا) دامنهٔ صفر دارند. ATR روی
    آن‌ها صفر می‌شود و هر نسبتی بر پایهٔ ATR انفجاری می‌شود. این کندل‌ها
    نه سیگنال می‌سازند و نه در آمار شمرده می‌شوند.
    """
    return (df.high.values - df.low.values) > min_range_pts * spec.point


def session_of(ts: pd.Series) -> pd.Series:
    """برچسب سشن بر پایهٔ ساعت UTC."""
    h = ts.dt.hour
    return pd.Series(
        np.select(
            [h < 6, h < 8, h < 12, h < 16, h < 21],
            ["ASIA", "LDN_OPEN", "LDN", "NY", "NY_PM"],
            default="LATE",
        ),
        index=ts.index,
    )


def profile(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """پروفایل ساعتی: حجم/دامنه/تعداد کندل برای فهم رفتار بازار."""
    d = df.copy()
    d["hour"] = d["time"].dt.hour
    d["minute"] = d["time"].dt.minute
    d["rng"] = d["high"] - d["low"]
    d["body"] = (d["close"] - d["open"]).abs()
    g = (
        d[d["minute"] == 0]
        .groupby("hour")
        .agg(bars=("close", "size"),
             mean_range_pts=("rng", lambda s: np.nan),
             med_range_pts=("rng", "median"),
             mean_volume=("volume", "mean"))
    )
    rng = d.groupby("hour")["rng"].median()
    g["mean_range_pts"] = rng.round(2)
    g["spread_pts"] = (d.groupby("hour")["spread"].median() / spec.point).round(1)
    # نسبت دامنه به اسپرد: مهم‌ترین معیار «آیا این ساعت برای اسکالپ اجازه می‌دهد؟»
    g["rng_over_spread"] = (g["mean_range_pts"] / g["spread_pts"].replace(0, np.nan)).round(2)
    g = g.drop(columns=["med_range_pts"])
    return g
