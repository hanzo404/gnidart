"""تشخیص ساختار بازار: سوئینگ، استخر نقدشوندگی، نقدشوندگی‌ربایی، FVG، شکست ساختار.

اصل حاکم بر این ماژول: **هیچ اطلاعاتی از آینده نباید وارد تصمیم امروز شود.**
همهٔ آشکارسازها با تأخیرِ تأیید (right-confirmation) نوشته شده‌اند؛ یعنی
سوئینگِ k-کندلی در کندل i+k تأیید می‌شود، نه در کندل i.

هر آشکارساز یک سیگنال در لحظهٔ «تأیید» تولید می‌کند؛ شبیه‌ساز تصمیم می‌گیرد
که آیا هنوز فرصت ورود باقی مانده یا نه.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────── اندیکاتورهای پایه

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    return pd.Series(true_range(df.high.values, df.low.values, df.close.values), index=df.index).ewm(
        alpha=1.0 / n, adjust=False
    ).mean().values


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    if len(x) == 0:
        return out
    a = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rolling_extreme(x: np.ndarray, n: int, fn) -> np.ndarray:
    """بیشینه/کمینه غلتان با پنجرهٔ n کندل (شامل خودِ کندل جاری)."""
    return pd.Series(x).rolling(n, min_periods=1).apply(fn, raw=True).values


# ───────────────────────────────────────────────────────── سوئینگ و نقدشوندگی

@dataclass
class StructureParams:
    swing_k: int = 3            # شعاع چپ/راست برای سوئینگ فراکتالی
    pool_tol_atr: float = 0.25  # تلورانس «هم‌سطح» بودن سقف/کف‌ها، بر حسب ATR
    sweep_max_bars: int = 1     # فاصلهٔ مجاز کندل بین شکست سطح و بازگشت
    displacement_atr: float = 0.8  # حداقل اندازهٔ بدنه برای «جابه‌جایی» (بر حسب ATR)
    fvg_min_atr: float = 0.10   # حداقل ارتفاع FVG بر حسب ATR
    mss_lookback: int = 12      # سقف/کف مرجع برای شکست ساختار
    m1_atr: int = 14


def swings(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """سوئینگ‌های تأییدشده. خروجی: آرایهٔ (نوع, قیمت, اندیس تأیید).

    نوع: +1 سقف، -1 کف، 0 هیچ. اندیس «تأیید» یعنی اندیس کندلی که از آن
    به بعد اطلاعاتش معتبر است (اندیس سوئینگ + k).
    """
    h, lo = df.high.values, df.low.values
    n = len(df)
    typ = np.zeros(n, dtype=np.int8)
    idx = np.zeros(n, dtype=np.int32)   # اندیس i که یک سوئینگ در i+k تأیید شده
    prc = np.zeros(n, dtype=np.float64)
    if n < 2 * k + 2:
        return typ, idx, prc
    # آینده‌نگری ممنوع است: برای کندل i، فقط می‌توانیم به i+k نگاه کنیم
    for i in range(k, n - k):
        window_h = h[i - k : i + k + 1]
        if h[i] >= window_h.max() and (window_h.argmax() == k):
            typ[i + k], idx[i + k], prc[i + k] = 1, i, h[i]
        window_l = lo[i - k : i + k + 1]
        if lo[i] <= window_l.min() and (window_l.argmin() == k):
            typ[i + k], idx[i + k], prc[i + k] = -1, i, lo[i]
    return typ, idx, prc


def liquidity_pools(typ: np.ndarray, prc: np.ndarray, tol: np.ndarray, window: int = 8) -> np.ndarray:
    """استخرهای نقدشوندگی (EQH/EQL) به‌صورت **علّی**: هر سطح فقط از سوئینگ‌هایی
    ساخته می‌شود که تا همان لحظه تأیید شده‌اند.

    ⚠️ نسخهٔ قبلی سوئینگ‌های *آینده* را هم در خوشه می‌آورد و آینده‌نگری
    می‌ساخت. اینجا برای هر کندل t، فقط `window` سوئینگِ تأییدشدهٔ اخیرِ همان
    جهت بررسی می‌شود؛ اگر دو یا بیشتر قیمتشان در tol به هم نزدیک باشند،
    میانگینشان «استخر» است.

    خروجی: آرایهٔ (n, 2) → [سطح بالایی، سطح پایینی] در هر لحظه (NaN اگر نبود).
    """
    n = len(typ)
    hi_lv = np.full(n, np.nan)
    lo_lv = np.full(n, np.nan)
    # فهرست سوئینگ‌های تأییدشده تا کنون، به تفکیک جهت
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(n):
        t = tol[i] if i < len(tol) else tol[-1]
        if t and t > 0:
            if typ[i] == 1:
                highs.append((i, prc[i]))
                if len(highs) > window:
                    highs.pop(0)
            elif typ[i] == -1:
                lows.append((i, prc[i]))
                if len(lows) > window:
                    lows.pop(0)
        if highs:
            last = highs[-1][1]
            grp = [p for _, p in highs if abs(p - last) <= t]
            if len(grp) >= 2:
                hi_lv[i] = float(np.mean(grp))
        if lows:
            last = lows[-1][1]
            grp = [p for _, p in lows if abs(p - last) <= t]
            if len(grp) >= 2:
                lo_lv[i] = float(np.mean(grp))
    return np.column_stack([hi_lv, lo_lv])


# ────────────────────────────────────────────────────── نقدشوندگی‌ربایی (Sweep)

def detect_sweeps(df: pd.DataFrame, p: StructureParams, atr_: np.ndarray) -> pd.DataFrame:
    """استخر نقدشوندگی جاری را می‌گیرد، سپس «شکست و بازگشت» را ثبت می‌کند.

    برای هر کندل t: آخرین سطح تأییدشده (قبل از t) را می‌گیریم؛ اگر
    low[t] < سطح (کف‌ها) ولی close[t] > سطح → نقدشوندگی‌ربایی نزولی (bearish sweep).
    خروجی فقط سیگنال‌هایی را دارد که در کندل t قابل دانستن‌اند.
    """
    typ, sidx, sprc = swings(df, p.swing_k)
    tol = atr_ * p.pool_tol_atr
    levels = liquidity_pools(typ, sprc, tol)
    lvl_hi, lvl_lo = levels[:, 0], levels[:, 1]
    out = []
    n = len(df)
    highs = df.high.values
    lows = df.low.values
    closes = df.close.values
    times = df.time.values

    for t in range(1, n):
        lh, ll = lvl_hi[t], lvl_lo[t]
        # جاروب نقدشوندگی: سقف را رد کرد و زیر آن بست
        if not np.isnan(lh) and highs[t] > lh and closes[t] < lh:
            out.append(dict(time=times[t], dir=-1, kind="sweep", level=lh,
                            extreme=highs[t], close=closes[t], atr=atr_[t], bar=t))
        if not np.isnan(ll) and lows[t] < ll and closes[t] > ll:
            out.append(dict(time=times[t], dir=1, kind="sweep", level=ll,
                            extreme=lows[t], close=closes[t], atr=atr_[t], bar=t))
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────── شکاف ارزش منصفانه

def detect_fvg(df: pd.DataFrame, p: StructureParams, atr_: np.ndarray) -> pd.DataFrame:
    """FVG سه‌کندلی: کف/سقف کندل سوم از سقف/کف کندل اول فاصله دارد.

    ناحیهٔ خالی = [high[t-2], low[t]] برای صعودی و [high[t], low[t-2]] برای نزولی.
    سیگنال در کندل t تأیید می‌شود.
    """
    h, lo = df.high.values, df.low.values
    n = len(df)
    out = []
    for t in range(2, n):
        if lo[t] > h[t - 2] and (lo[t] - h[t - 2]) >= p.fvg_min_atr * atr_[t]:
            out.append(dict(time=df.time.values[t], dir=1, top=lo[t], bot=h[t - 2], atr=atr_[t], bar=t))
        if h[t] < lo[t - 2] and (lo[t - 2] - h[t]) >= p.fvg_min_atr * atr_[t]:
            out.append(dict(time=df.time.values[t], dir=-1, top=lo[t - 2], bot=h[t], atr=atr_[t], bar=t))
    return pd.DataFrame(out)


def displacement(df: pd.DataFrame, p: StructureParams, atr_: np.ndarray) -> np.ndarray:
    """کندل‌های جابه‌جایی: بدنهٔ بزرگ در جهت خودِ کندل (مومنتوم واقعی)."""
    body = (df.close.values - df.open.values)
    return (np.abs(body) >= p.displacement_atr * atr_) * np.sign(body)
