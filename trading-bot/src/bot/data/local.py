"""لودر و اعتبارسنجی CSVهای OHLCV محلی (خروجی MT5 یا Dukascopy).

دو اسکیمای رایج را می‌شناسد:
    MT5 export : time, open, high, low, close, tick_volume, spread
    Duka CSV   : time, open, high, low, close, volume

اعتبارسنجی (گیت کیفیت داده قبل از هر بک‌تست):
    - تکراری زمان (drop + گزارش)
    - high < low یا open/close بیرون [low, high]
    - قیمت‌های غیرمثبت
    - پرش‌های بزرگ زمانی (گپ > ۳ روز)
    - آمار کلی: تعداد، بازه، روزهای هفته حاضر
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

_ALIASES = {
    "tick_volume": "volume",
    "vol": "volume",
    "datetime": "time",
    "date": "time",
    "timestamp": "time",
}
_REQUIRED = ["time", "open", "high", "low", "close"]


def load_ohlc_csv(path: str | Path) -> Tuple[pd.DataFrame, Dict]:
    """برمی‌گرداند: (df تمیزشده, گزارش کیفیت). حجم اختیاری است."""
    p = Path(path)
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=_ALIASES)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"ستون‌های لازم غایب‌اند: {missing} (ستون‌های فایل: {list(df.columns)})")

    # زمان: اول ISO، بعد فرمت MT5 (2026.09.08 22:23)
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.isna().any():
        t2 = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M", errors="coerce")
        t = t.fillna(t2)
    if t.isna().any():
        raise ValueError(f"{int(t.isna().sum())} ردیف زمان نامعتبر دارد")
    df["time"] = t

    keep = _REQUIRED + (["volume"] if "volume" in df.columns else [])
    df = df[keep].copy()

    dup = int(df.duplicated(subset="time").sum())
    if dup:
        df = df.drop_duplicates(subset="time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    report = validate(df)
    report["duplicates_dropped"] = dup
    report["source"] = p.name
    return df, report


def validate(df: pd.DataFrame) -> Dict:
    """گزارش کیفیت — قبل از بک‌تست چک شود: violations باید ~0 باشد."""
    n = len(df)
    rep: Dict = {"rows": n}
    if n == 0:
        return rep
    rep["start"] = str(df["time"].iloc[0])
    rep["end"] = str(df["time"].iloc[-1])

    bad_hl = int((df["high"] < df["low"]).sum())
    bad_oc = int(((df["open"] < df["low"]) | (df["open"] > df["high"]) |
                  (df["close"] < df["low"]) | (df["close"] > df["high"])).sum())
    bad_px = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

    gaps = df["time"].diff().dt.total_seconds() / 86400.0
    big_gaps = int((gaps > 3.0).sum())
    weekdays = sorted(df["time"].dt.dayofweek.unique().tolist())  # 0=دوشنبه

    rep.update({
        "ohlc_violations": bad_hl + bad_oc,
        "nonpositive_prices": bad_px,
        "gaps_gt_3d": big_gaps,
        "weekdays_present": weekdays,
        "price_min": float(df["low"].min()),
        "price_max": float(df["high"].max()),
    })
    if "volume" in df.columns:
        rep["volume_total"] = float(df["volume"].sum())
    return rep
