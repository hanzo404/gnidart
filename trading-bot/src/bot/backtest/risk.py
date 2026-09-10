"""لایهٔ ریسک بک‌تست — فاز ۳: اتصال نردبان بریکر و سایزینگ %-ریسک.

این ماژول همان CircuitBreaker فاز ۰ (تست‌شده) را به موتور بک‌تست وصل
می‌کند؛ هیچ منطق جدیدی اختراع نمی‌شود:

    سایز = equity × risk_pct × ضریب بریکر ÷ فاصله تا استاپ
    گیت ورود = allow_entry (توقف/مکث/فقط-A)
    دراودان ماهانه −۶٪ → halt تا پایان ماه (auto-ack در مرز ماه —
    در بک‌تست انسانی برای acknowledge نیست؛ در دموی زنده، acknowledge
    فقط بعد از بازبینی walk-forward)

انتخاب صادقانه و مستند: تا وقتی سیستمِ درجه‌بندی ستاپ (A/B/C) نداریم،
استراتژیِ فعلیِ واحد را 'A' حساب می‌کنیم → deep-derate از راه «حجم»
(×۰.۲۵) خودش را نشان می‌دهد نه بن‌بست ورود. (اگر 'B' می‌بود، بعد از
اولین رشتهٔ ۳-باختی — که با نرخ برد ~۳۳٪ خیلی زود می‌آید — همهٔ ورودها
برای همیشه بلاک می‌شد؛ بردِ بازگرداننده هرگز نمی‌آمد.) وقتی درجه‌بندی
پیاده شود، این گیت خودبه‌خود سخت‌گیرتر می‌شود.

نکتهٔ دنیای واقعی (لبهٔ گرانولاریتی لات): روی حساب $3k با ریسک ۰.۵٪،
سایز واقعی ~0.01 لات است؛ کاهش بریکر به ×۰.۲۵ از کفِ لاتِ بروکر
پایین‌تر نمی‌رود و به کف clamps می‌شود. موتور این را می‌شمارد
(min_lot_clamps) تا در گزارش صادق باشیم.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from bot.risk.circuit_breaker import CircuitBreaker


class RiskPolicy(Protocol):
    """قرارداد موتور بک‌تست با هر سیاست ریسک (بریکر، سقف روزانه، ...)."""

    def allow_entry(self, ts) -> bool: ...
    def size_multiplier(self) -> float: ...
    def on_trade_closed(self, r: float, ts, equity: float) -> None: ...
    def on_bar_close(self, ts, equity: float) -> None: ...


class BreakerPolicy:
    """CircuitBreaker + ردیابی ماهانه + شمارنده‌های رویداد."""

    def __init__(self, breaker: CircuitBreaker | None = None,
                 grade: str = "A", auto_ack_monthly: bool = True):
        self.breaker = breaker or CircuitBreaker()
        self.grade = grade
        self.auto_ack_monthly = auto_ack_monthly
        self._month: str | None = None
        self._month_start_eq: float | None = None
        self.counters: dict[str, int] = {}
        self.halted_months: list[str] = []

    # ------------------------------------------------------------------ #
    def _bump(self, kind: str) -> None:
        self.counters[kind] = self.counters.get(kind, 0) + 1

    def allow_entry(self, ts) -> bool:
        ts = pd.Timestamp(ts)
        ok, _ = self.breaker.allow_entry(self.grade, ts)
        if not ok:
            self._bump("entry_blocks")
        return ok

    def size_multiplier(self) -> float:
        return self.breaker.size_multiplier()

    def on_trade_closed(self, r: float, ts, equity: float) -> list:
        out = self.breaker.on_trade_closed(r, pd.Timestamp(ts))
        for ev in out:
            self._bump(ev.kind)
        return out

    def on_bar_close(self, ts, equity: float) -> None:
        ts = pd.Timestamp(ts)
        month = ts.strftime("%Y-%m")
        if month != self._month:
            # مرز ماه: اگر halt شده، بازبینی ماهانه = auto-ack (انتخاب مستند)
            if self.auto_ack_monthly and self.breaker.halted:
                if self.breaker.acknowledge(ts):
                    self._bump("ack")
                    self.halted_months.append(self._month or month)
            self._month = month
            self._month_start_eq = equity
            return
        if self._month_start_eq:
            dd = equity / self._month_start_eq - 1.0
            if dd <= -self.breaker.max_monthly_dd:
                if self.breaker.on_monthly_drawdown(dd, ts):
                    self._bump("monthly_halt")
                    self.halted_months.append(month)
