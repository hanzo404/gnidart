"""ساخت ستاپ: زنجیرهٔ کامل ICT/SMC به‌صورت الگوریتمی و پارامتریک.

زنجیره (هر حلقه اختیاری است تا بتوان «کدام تکه واقعاً ارزش دارد» را اندازه گرفت):

    نقدشوندگی‌ربایی  →  جابه‌جایی (displacement)  →  شکست ساختار (MSS)
                     →  شکاف ارزش منصفانه (FVG)  →  ورود لیمیت  →  استاپ ساختاری

هدف طراحی: هر پارامتر یک عددِ «قابل خاموش کردن» باشد تا بتوان سود هر جزء
را جدا اندازه گرفت. استراتژی نهایی فقط از اجزایی ساخته می‌شود که در
آزمایش سودِ افزودهٔ مثبت داشته باشند.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .structure import atr, ema, swings


@dataclass
class SetupParams:
    # ── سوگیری تایم بالا (زمینهٔ جهت)
    use_bias: bool = True
    bias_fast: int = 20        # کندل M5 → ۱۰۰ دقیقه
    bias_slow: int = 48         # ۴ ساعت
    # ── جابه‌جایی
    disp_atr: float = 0.8
    disp_max_bars: int = 3
    # ── شکست ساختار
    require_mss: bool = True
    mss_lookback: int = 10
    mss_max_bars: int = 4
    # ── FVG
    require_fvg: bool = True
    fvg_lookback: int = 5
    fvg_min_atr: float = 0.15
    fvg_entry: float = 0.5     # ۰ = لبهٔ نزدیک، ۱ = لبهٔ دور
    require_fvg_unfilled: bool = True
    # ── ریسک ساختاری
    stop_buffer_atr: float = 0.15
    min_stop_atr: float = 0.30
    max_stop_atr: float = 3.0
    min_stop_spread: float = 4.0
    # ── زمان
    allowed_hours: tuple = ()
    setup_max_age: int = 4     # حداکثر فاصلهٔ ورود از لحظهٔ جاروب (کندل M5)

    def tag(self) -> str:
        return (
            f"bias{int(self.use_bias)}_d{self.disp_atr}_mss{int(self.require_mss)}"
            f"_fvg{int(self.require_fvg)}_h{'-'.join(map(str, self.allowed_hours)) or 'all'}"
        )


def build_bias(df: pd.DataFrame, p: SetupParams) -> np.ndarray:
    """سوگیری جهت: +1 صعودی، -1 نزولی، 0 خنثی."""
    if not p.use_bias:
        return np.zeros(len(df), dtype=np.int8)
    c = df.close.values
    fast, slow = ema(c, p.bias_fast), ema(c, p.bias_slow)
    bias = np.zeros(len(df), dtype=np.int8)
    bias[(fast > slow) & (c > fast)] = 1
    bias[(fast < slow) & (c < fast)] = -1
    return bias


def build_signals(
    m5: pd.DataFrame,
    sweeps: pd.DataFrame,
    atr_: np.ndarray,
    p: SetupParams,
    point: float,
    spread: float,
) -> pd.DataFrame:
    """از فهرست جاروب‌ها، ستاپ‌های کامل و معتبر را می‌سازد.

    خروجی: time (لحظهٔ تأیید = کندل شکست ساختار)، dir, entry (لیمیت)، stop
    """
    if sweeps.empty:
        return pd.DataFrame(columns=["time", "dir", "entry", "stop", "tag"])

    h, lo, c, o = m5.high.values, m5.low.values, m5.close.values, m5.open.values
    n = len(m5)
    bias = build_bias(m5, p)
    typ, sidx, sprc = swings(m5, 3)
    # بالاترین/پایین‌ترین قیمت n کندل گذشته (برای تشخیص شکست ساختار)
    roll_hi = pd.Series(h).rolling(p.mss_lookback, min_periods=2).max().shift(1).values
    roll_lo = pd.Series(lo).rolling(p.mss_lookback, min_periods=2).min().shift(1).values
    hours = m5.time.dt.hour.values

    rows = []
    for sw in sweeps.itertuples(index=False):
        t, d = int(sw.bar), int(sw.dir)
        if t + p.setup_max_age >= n or atr_[t] <= 0:
            continue
        if p.allowed_hours and hours[t] not in p.allowed_hours:
            continue
        a = atr_[t]
        if p.use_bias and bias[t] != 0 and bias[t] != d:
            continue  # خلاف جهت زمینهٔ تایم بالا = حذف

        # ۱) جابه‌جایی در جهت برگشت
        disp_bar = -1
        for k in range(0, p.disp_max_bars + 1):
            b = t + k
            if b >= n:
                break
            body = c[b] - o[b]
            if body * d >= p.disp_atr * atr_[b]:
                disp_bar = b
                break
        if disp_bar < 0:
            continue

        # ۲) شکست ساختار در جهت برگشت
        mss_bar = -1
        if p.require_mss:
            ref = roll_hi[t] if d == 1 else roll_lo[t]
            for k in range(disp_bar, min(disp_bar + p.mss_max_bars + 1, t + p.setup_max_age + 1)):
                if d == 1 and c[k] > ref:
                    mss_bar = k
                    break
                if d == -1 and c[k] < ref:
                    mss_bar = k
                    break
            if mss_bar < 0:
                continue
        else:
            mss_bar = disp_bar

        # ۳) FVG در جهت جابه‌جایی، ترجیحاً هنوز پرنشده
        entry = np.nan
        for k in range(disp_bar, mss_bar + 1):
            if k + 2 >= n:
                continue
            if d == 1 and lo[k] > h[k - 2] and (lo[k] - h[k - 2]) >= p.fvg_min_atr * atr_[k]:
                top, bot = lo[k], h[k - 2]
            elif d == -1 and h[k] < lo[k - 2] and (lo[k - 2] - h[k]) >= p.fvg_min_atr * atr_[k]:
                top, bot = lo[k - 2], h[k]
            else:
                continue
            if p.require_fvg_unfilled:
                seg_lo, seg_hi = lo[k + 1:mss_bar + 1], h[k + 1:mss_bar + 1]
                if len(seg_lo) and d == 1 and seg_lo.min() <= bot:
                    continue
                if len(seg_lo) and d == -1 and seg_hi.max() >= top:
                    continue
            entry = bot + (top - bot) * p.fvg_entry
            break
        if p.require_fvg and not np.isfinite(entry):
            continue
        if not np.isfinite(entry):
            # بدون FVG: ورود لیمیت روی ۵۰٪ پایهٔ حرکت جابه‌جایی
            base_hi, base_lo = h[disp_bar], lo[disp_bar]
            entry = base_lo + (base_hi - base_lo) * 0.5

        # ۴) استاپ پشت extremum جاروب + بافر
        stop = sw.extreme - d * p.stop_buffer_atr * a
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        if risk < p.min_stop_atr * a or risk > p.max_stop_atr * a:
            continue
        if risk < p.min_stop_spread * spread:
            continue
        rows.append((m5.time.values[mss_bar], d, float(entry), float(stop), p.tag()))

    return pd.DataFrame(rows, columns=["time", "dir", "entry", "stop", "tag"])
