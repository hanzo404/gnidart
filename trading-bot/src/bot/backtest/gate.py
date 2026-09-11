"""گیت رژیم/سشن — فاز ۲: سیگنال فقط با اجازهٔ رژیم و سشن عبور می‌کند.

نقش: استراتژیِ پایه (مثلاً v0) سیگنال می‌دهد؛ این لایه تصمیم می‌گیرد
«الان، این جهت» مجاز است یا نه:
    - CHAOS → هیچ ورودی
    - خرید فقط در TREND_UP، فروش فقط در TREND_DOWN
    - خارج از سشن ورود (پیش‌فرض ۱۲–۲۰ UTC) → هیچ ورودی
خروج (استاپ/تارگت) هرگز بلاک نمی‌شود — مدیریت ریسک همیشه روشن است.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from bot.backtest.engine import Order, Strategy
from bot.regime.engine import CHAOS, TREND_DOWN, TREND_UP


class GatedStrategy:
    def __init__(self, inner: Strategy, regime: np.ndarray,
                 session_open_mask: np.ndarray,
                 direction: int | None = None):
        """direction: None=هر دو | +1 فقط خرید | -1 فقط فروش.

        نمونهٔ کاربرد: وقتی لایهٔ تشخیص/پست‌مورتم شواهد داد که یک جهت
        خالص‌زیان‌ده است، حذف آن یک «آزمایش sandbox» است — نه تغییر
                        خودجوش. باید از walk-forward و تأیید انسانی رد شود."""
        self.inner = inner
        self.regime = np.asarray(regime)
        self.sess = np.asarray(session_open_mask, dtype=bool)
        self.direction = direction

    def prepare(self, bars) -> None:
        self.inner.prepare(bars)

    def on_bar(self, i: int) -> Optional[Order]:
        # ورود در open کندل بعدی رخ می‌دهد → سشنِ همان کندل ملاک است
        if not self.sess[min(i + 1, len(self.sess) - 1)]:
            return None
        r = self.regime[i]          # رژیمِ لحظهٔ تصمیم (ضد-نشتی)
        if r == CHAOS:
            return None
        sig = self.inner.on_bar(i)
        if sig is None:
            return None
        if self.direction is not None and sig.direction != self.direction:
            return None
        if sig.direction > 0 and r != TREND_UP:
            return None
        if sig.direction < 0 and r != TREND_DOWN:
            return None
        return sig
