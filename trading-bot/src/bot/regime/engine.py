"""موتور رژیم — فاز ۲: تشخیص «کی باید معامله کرد و در چه جهتی».

سه سیگنال مستقل و تفسیرپذیر (نه جعبهٔ سیاه):
    1. ADX(14) وایلدر — قدرت روند (با هیسترزیس: ورود ≥ ۲۳، خروج < ۲۰
       تا از نوسان مرزی عقب‌وجلو نکند)
    2. جهت — شیب EMA50 (۲۰ کندل اخیر) + موقعیت قیمت نسبت به EMA
    3. CHAOS — ATR سری (۱۴) بیش از ~۱.۸ برابرِ ATR کند (۲۰۰) =
       انفجار نوسان لحظه‌ای. مقیاس‌مستقل (نسبت، نه سطح) و مستقل از
       جهت — در روند نزولیِ سالم آتش نمی‌زند، در شوک خبری می‌زند.

خروجی هر کندل: TREND_UP=+1 | TREND_DOWN=−1 | RANGE=0 | CHAOS=2

قرارداد علیّت: رژیم کندل i فقط از کندل‌های ≤ i محاسبه می‌شود —
هیچ شیفت/پنجرهٔ آینده‌ای در کار نیست (تست علّیت در tests/test_regime.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TREND_UP, TREND_DOWN, RANGE, CHAOS = 1, -1, 0, 2
NAMES = {TREND_UP: "TREND_UP", TREND_DOWN: "TREND_DOWN",
         RANGE: "RANGE", CHAOS: "CHAOS"}


@dataclass
class RegimeConfig:
    adx_period: int = 14
    adx_enter: float = 23.0     # هیسترزیس: ورود به روند
    adx_exit: float = 20.0      # هیسترزیس: خروج از روند
    ema_period: int = 50
    slope_lookback: int = 20    # شیب EMA روی این تعداد کندل
    atr_period: int = 14
    chaos_slow_period: int = 200
    chaos_ratio: float = 1.8    # ATR14/ATR200 بالاتر از این = CHAOS


def _wilder_ewm(x: np.ndarray, period: int) -> np.ndarray:
    """هموارسازی وایلدر = EWM با alpha=1/period."""
    return (pd.Series(x).ewm(alpha=1.0 / period, adjust=False).mean()
            .to_numpy())


def wilder_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray,
               period: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """ADX وایلدر + DIها (برداری؛ فقط دادهٔ گذشته)."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
    atr = _wilder_ewm(tr, period)
    pdi = 100.0 * _wilder_ewm(plus_dm, period) / (atr + 1e-12)
    mdi = 100.0 * _wilder_ewm(minus_dm, period) / (atr + 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    return _wilder_ewm(dx, period), atr


class RegimeEngine:
    def __init__(self, cfg: RegimeConfig | None = None):
        self.cfg = cfg or RegimeConfig()

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """کندل‌ها (M15، هر تایم‌زونی) → DataFrame رژیم هم‌طول با bars."""
        cfg = self.cfg
        c = bars["close"].to_numpy(float)
        h = bars["high"].to_numpy(float)
        l = bars["low"].to_numpy(float)
        n = len(bars)

        adx, atr = wilder_adx(h, l, c, cfg.adx_period)
        atr_slow = _wilder_ewm(atr, cfg.chaos_slow_period)
        # نسبت ATR سری به کند: انفجار نوسان، مستقل از سطح قیمت و جهت
        chaos = atr > cfg.chaos_ratio * atr_slow
        atr_pct = atr / (c + 1e-12)

        ema = pd.Series(c).ewm(span=cfg.ema_period, adjust=False).mean().to_numpy()
        slope = (ema - np.roll(ema, cfg.slope_lookback)) / (c + 1e-12) * 1e4  # bps
        slope[:cfg.slope_lookback] = 0.0

        up_raw = (adx >= cfg.adx_enter) & (slope > 0) & (c > ema)
        dn_raw = (adx >= cfg.adx_enter) & (slope < 0) & (c < ema)

        labels = np.zeros(n, dtype=np.int8)
        state = RANGE
        for i in range(n):
            if chaos[i]:
                state = CHAOS
            elif state == TREND_UP:
                if dn_raw[i]:
                    state = TREND_DOWN
                elif adx[i] < cfg.adx_exit or c[i] < ema[i]:
                    state = RANGE
            elif state == TREND_DOWN:
                if up_raw[i]:
                    state = TREND_UP
                elif adx[i] < cfg.adx_exit or c[i] > ema[i]:
                    state = RANGE
            else:  # RANGE یا تازه از CHAOS آمده
                state = RANGE
                if up_raw[i]:
                    state = TREND_UP
                elif dn_raw[i]:
                    state = TREND_DOWN
            labels[i] = state

        return pd.DataFrame({
            "time": bars["time"].to_numpy(),
            "regime": labels, "adx": adx, "atr_pct": atr_pct,
            "ema50": ema, "slope_bps": slope,
        })


def session_open(times: pd.Series | np.ndarray,
                 start_hour: int = 12, end_hour: int = 20) -> np.ndarray:
    """ماسک سشن ورود (ساعت UTC). پیش‌فرض ۱۲:۰۰–۱۹:۵۹ = هم‌پوشانی لندن/نیویورک.

    ۲۱:۰۰ UTC (رول‌اور/بدترین اسپرد $0.18) و ساعات کم‌جان آسیا بیرون می‌مانند.
    فقط «ورود» را گیت می‌کند — استاپ و تارگت همیشه فعال‌اند.
    """
    hours = pd.Series(times).dt.hour.to_numpy()
    return (hours >= start_hour) & (hours < end_hour)
