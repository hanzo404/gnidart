"""آماده‌سازی داده برای بک‌تست فاز ۱ — دو منبع، یک استاندارد.

وظایف این ماژول:
    1. بارگذاری دو منبع:
       - Dukascopy (M1، BID، تایم‌استامپ UTC، با کندل‌های پدینگ آخر هفته/سِشن بسته)
       - MT5 (M1، تایم‌استامپ سرور بروکر، با ستون spread بر حسب پوینت)
    2. فیلتر کندل‌های پدینگ duka (حجم صفر + OHLC تخت = بازار بسته)
    3. تشخیص تجربی اختلاف ساعت سرور MT5 با UTC (جستجوی آفست که کمترین
       اختلاف قیمت با duka را می‌دهد — ماه‌به‌ماه تا تغییرات DST هم دیده شود)
    4. ریسمپل M1 → M15 با اسپرد «اولین دقیقه» (ورود ما در ابتدای کندل است)
    5. پروفایل اسپرد ساعتی از MT5 → چسباندن به کندل‌های duka (که اسپرد ندارند)

قرارداد خروجی همهٔ توابع: DataFrame با ستون‌های
    time (datetime64, tz-naive) / open / high / low / close / volume
    و در صورت وجود: spread_usd (دلار به ازای هر اونس)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["time", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------- #
# بارگذاری منابع
# --------------------------------------------------------------------- #
def load_duka(path: str | Path) -> pd.DataFrame:
    """Dukascopy M1 → حذف کندل‌های پدینگ (حجم ۰ و OHLC کاملاً تخت).

    این کندل‌ها لحظات بستهٔ بازار (آخر هفته/سِشن تعطیل) هستند؛ اگر نمانند
    اندیکاتورها و تحلیل سشن به‌کلی خراب می‌شوند (~۳۲٪ ردیف‌های duka!).
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df[COLS].copy()
    flat = ((df["volume"] == 0)
            & (df["open"] == df["close"])
            & (df["open"] == df["high"])
            & (df["open"] == df["low"]))
    df = df[~flat].reset_index(drop=True)
    return df


def load_mt5(path: str | Path, point: float = 0.01) -> pd.DataFrame:
    """MT5 M1 → ستون‌های استاندارد + spread_usd.

    spread خام در MT5 بر حسب «پوینت» است؛ برای XAUUSD هر پوینت = $0.01
    → ۱۴ پوینت = $0.14 spread به ازای هر اونس.
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns={"tick_volume": "volume"})
    df["volume"] = df["volume"].astype(float)
    if point == 0.01 and len(df) and float(df["close"].median()) < 500:
        # ممیزی خارجی: پیش‌فرضِ طلایی روی نماد ارزان‌تر = اسپرد ۱۰ برابری غلط
        print(f"⚠️ load_mt5 با point=0.01 روی میانهٔ قیمت "
              f"{float(df['close'].median()):.2f} — اگر نماد XAGUSD است "
              f"point=0.001 بده")
    df["spread_usd"] = df["spread"] * point
    return df[COLS + ["spread_usd"]].reset_index(drop=True)


# --------------------------------------------------------------------- #
# اختلاف ساعت سرور MT5 با UTC
# --------------------------------------------------------------------- #
def _month_key(t: pd.Series) -> pd.Series:
    return t.dt.strftime("%Y-%m")


def infer_offset_minutes(mt5: pd.DataFrame, duka: pd.DataFrame,
                         candidates=range(0, 241, 15)) -> dict[str, int]:
    """ماه‌به‌ماه: کدام شیفت (دقیقه) قیمت MT5 را روی UTC انداخته باشد؟

    منطق: هر دو منبع BID یک بازارند → در آفست درست، اختلاف close هم‌دقیقه‌ها
    باید چند سنت باشد؛ در آفست غلط، چند دلار. خروجی: {'2024-01': 120, ...}
    """
    # نکته: pandas 3 ممکن است datetime64[us] بدهد — با timedelta64 نوع‌امن کار می‌کنیم
    m_time = mt5["time"].to_numpy()
    m_close = mt5["close"].to_numpy()
    d_time = duka["time"].to_numpy()
    d_close = duka["close"].to_numpy()
    keys = _month_key(mt5["time"]).to_numpy()   # یک‌بار — نه در حلقه (سریع‌تر ۳۰x)
    months = sorted(set(keys))
    out: dict[str, int] = {}
    for mo in months:
        mask = keys == mo
        mt_m, mt_c = m_time[mask], m_close[mask]
        if len(mt_m) < 500:        # ماه ناقص → رد کن
            continue
        best_off, best_med = None, np.inf
        for off in candidates:
            shifted = mt_m - np.timedelta64(off, "m")
            idx = np.searchsorted(d_time, shifted)
            idx.clip(0, len(d_time) - 1, out=idx)
            hit = d_time[idx] == shifted
            if hit.sum() < 500:
                continue
            diff = np.abs(mt_c[hit] - d_close[idx][hit])
            med = float(np.median(diff))
            if med < best_med:
                best_med, best_off = med, off
        if best_off is not None:
            out[mo] = int(best_off)
    return out


def shift_mt5_to_utc(mt5: pd.DataFrame, offsets: dict[str, int]) -> pd.DataFrame:
    """تایم‌استامپ سرور → UTC با آفست ماه‌انه (DST خودکار هندل می‌شود)."""
    keys = _month_key(mt5["time"]).map(
        lambda k: offsets.get(k, sorted(offsets.values())[0] if offsets else 0))
    shifted = mt5["time"] - pd.to_timedelta(keys, unit="m")
    out = mt5.copy()
    out["time"] = shifted
    return out.sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------- #
# ریسمپل M1 → M15 (با اسپرد اولین دقیقه)
# --------------------------------------------------------------------- #
def to_m15(m1: pd.DataFrame) -> pd.DataFrame:
    """M1 → M15. open=اول، high=max، low=min، close=آخر، volume=جمع،
    spread_usd=اسپردِ اولین دقیقه (ورود ما ابتدای کندل اتفاق می‌افتد)."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    if "spread_usd" in m1.columns:
        agg["spread_usd"] = "first"
    out = (m1.set_index("time")
           .resample("15min", label="left", closed="left")
           .agg(agg)
           .dropna(subset=["open"])
           .reset_index())
    return out


def resample_tf(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """ریسمپل عمومی (برای H4 بایاس و ...). rule مثل '15min' یا '4h'."""
    out = (m1.set_index("time")
           .resample(rule, label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna(subset=["open"])
           .reset_index())
    return out


# --------------------------------------------------------------------- #
# پروفایل اسپرد ساعتی (از MT5) → کندل‌های duka
# --------------------------------------------------------------------- #
def spread_profile_by_hour(mt5_utc_m1: pd.DataFrame) -> pd.Series:
    """میانهٔ اسپرد بر حسب ساعت UTC (اینديکس 0..23)."""
    g = mt5_utc_m1.groupby(mt5_utc_m1["time"].dt.hour)["spread_usd"]
    return g.median()


def attach_spread_profile(bars: pd.DataFrame, profile: pd.Series) -> pd.DataFrame:
    """به کندل‌های بدون اسپرد (duka)، اسپردِ میانهٔ همان ساعت را بچسبان."""
    out = bars.copy()
    hours = out["time"].dt.hour
    out["spread_usd"] = hours.map(profile).astype(float).to_numpy()
    out["spread_usd"] = out["spread_usd"].fillna(float(profile.median()))
    return out
