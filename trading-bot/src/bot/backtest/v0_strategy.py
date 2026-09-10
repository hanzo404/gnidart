"""پورتِ بک‌تستِ استراتژی v0 (شاخهٔ آرشیو hanzo404-patch-1) — وفادار اما علّی.

منطق v0 دقیقاً همان بود (fast_execution.py + ai_brain.py + smc_logic.py):
    - تریگر: FVG سه‌کندلی — صعودی اگر low[i] > high[i-2]، نزولی برعکس
    - بایاس: ساختار H4 — اگر close آخرین H4 بالای سقف ۹ کندل قبل → BULLISH،
      زیر کف → BEARISH، وگرنه NEUTRAL (هر دو جهت مجاز)
    - SL: کف/سقف ۵ کندل اخیر ∓ $0.50 | TP: ۲.۵ برابر ریسک
    - سایز: ثابت 0.01 لات | کول‌داون: ۳۰۰ ثانیه

دو تفاوت عمدی و مستند با کد اصلی (رفع باگ، نه تغییر استراتژی):
    1. بایاس فقط از H4 «بسته‌شده» محاسبه می‌شود — v0 روی کندل در حال
       تشکیل H4 تحلیل می‌کرد (repaint؛ بایاس هر ۳۰ دقیقه عوض می‌شد).
    2. پر شدن در open کندل بعدی — v0 در چند ثانیه بعدِ سیگنال وارد می‌شد؛
       open کندل بعدی نزدیک‌ترین تقریب بدون نشتی است.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from bot.backtest.engine import Order, Strategy


class V0Strategy:
    """استراتژی v0 روی هر تایم‌فریمی از کندل‌ها (M1 وفادار / M15 ترجمه‌شده).

    bias_h4: DataFrame کندل‌های H4 همان منبع (ستون‌های time/high/low/close).
    """

    def __init__(self, bias_h4: pd.DataFrame, sl_pad: float = 0.50,
                 rr: float = 2.5, min_bars: int = 5):
        self.sl_pad = sl_pad
        self.rr = rr
        self.min_bars = min_bars
        # بایاس روی H4 بسته‌شده: close کندل j در برابر سقف/کف ۹ کندل قبلش
        h4 = bias_h4.reset_index(drop=True)
        n = len(h4)
        labels = np.zeros(n, dtype=np.int8)  # 0=NEUTRAL, +1=BULLISH, -1=BEARISH
        if n >= 10:
            hi = h4["high"].to_numpy(float)
            lo = h4["low"].to_numpy(float)
            cl = h4["close"].to_numpy(float)
            hi9 = np.full(n, np.nan)
            lo9 = np.full(n, np.nan)
            for j in range(9, n):  # فقط ~۵ هزار کندل H4 در ۳ سال — سریع
                hi9[j] = hi[j - 9:j].max()
                lo9[j] = lo[j - 9:j].min()
            labels[cl > hi9] = 1
            labels[cl < lo9] = -1
        # بایاس از «پایان» کندل H4 معتبر است (time = برچسب ابتدای کندل)
        if n:
            self._bias_t = h4["time"].to_numpy() + np.timedelta64(4, "h")
            self._bias_v = labels
        else:
            self._bias_t = np.array([], dtype="datetime64[ns]")
            self._bias_v = labels

    # ------------------------------------------------------------------ #
    def prepare(self, bars: pd.DataFrame) -> None:
        self._n = len(bars)
        low = bars["low"].to_numpy(float)
        high = bars["high"].to_numpy(float)
        # FVG سه‌کندلی (فقط دادهٔ گذشته: کندل i در برابر i-2)
        self._bull = np.zeros(self._n, dtype=bool)
        self._bear = np.zeros(self._n, dtype=bool)
        self._bull[2:] = low[2:] > high[:-2]
        self._bear[2:] = high[2:] < low[:-2]
        # SL: کف/سقف ۵ کندل اخیر (شامل کندل سیگنال) ∓ pad
        lows5 = pd.Series(low).rolling(5).min().to_numpy()
        highs5 = pd.Series(high).rolling(5).max().to_numpy()
        self._sl_buy = lows5 - self.sl_pad
        self._sl_sell = highs5 + self.sl_pad
        # بایاس هر کندل در «زمان close» آن (ضد repaint)
        close_t = bars["time"].to_numpy() + self._bar_dt(bars)
        if len(self._bias_v):
            idx = np.searchsorted(self._bias_t, close_t, side="right") - 1
            valid = idx >= 0
            idxc = np.clip(idx, 0, len(self._bias_v) - 1)
            bias = np.where(valid, self._bias_v[idxc], 0)
        else:
            bias = np.zeros(self._n)
        self._bias = bias.astype(np.int8)

    @staticmethod
    def _bar_dt(bars: pd.DataFrame) -> np.timedelta64:
        """فاصلهٔ استاندارد کندل‌ها — کوچک‌ترین فاصلهٔ مثبت (ضد گپ آخر هفته)."""
        if len(bars) < 2:
            return np.timedelta64(1, "m")
        d = np.diff(bars["time"].to_numpy())   # timedelta64 — نوع‌امن
        d = d[d > np.timedelta64(0)]
        return d.min() if len(d) else np.timedelta64(1, "m")

    # ------------------------------------------------------------------ #
    def on_bar(self, i: int) -> Optional[Order]:
        if i < self.min_bars:
            return None
        b = self._bias[i]
        if self._bull[i] and b >= 0:      # BULLISH یا NEUTRAL (مثل v0)
            return Order(+1, float(self._sl_buy[i]), self.rr, "v0_fvg")
        if self._bear[i] and b <= 0:
            return Order(-1, float(self._sl_sell[i]), self.rr, "v0_fvg")
        return None
