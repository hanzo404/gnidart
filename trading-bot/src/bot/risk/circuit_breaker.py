"""نردبان برق‌گیر ضرر (Circuit Breaker) — ماژول خویش‌تن‌داری ربات.

چرا این ماژول وجود دارد؟
    حتی یک سیستم سودده با نرخ برد ۵۵٪:
      - ۲۰٪ مواقع دو باخت پشت‌سرهم می‌خورد (P = 0.45² ≈ 0.20)
      - در هر ۱۰۰ معامله تقریباً به‌طور قطعی یک رشته ۴-باختی دارد
    واکنش غلط (بازنویسی استراتژی / افزایش حجم برای جبران) = قتلِ لبه.
    واکنش درست: کاهش پله‌ای ریسک + ثبت کانتکست + تحلیل آماری در نمونه بزرگ.

نردبان (همه پارامترها از کانفیگ):
    ضرر متوالی ۲  → حجم ×۰.۵  + صف اسکن تشخیصی (regime/spread/خبر)
    ضرر متوالی ۳  → حجم ×۰.۲۵ + فقط ستاپ‌های A-grade
    ضرر متوالی ۴  → توقف `pause_hours` ساعت + پست‌مورتم اجباری
    ضرر متوالی ۶  → توقف کامل؛ ادامه فقط با acknowledge() (بازاجرای walk-forward)
    دراودان ماهانه ≥ max_monthly_dd → توقف کامل تا acknowledge

بازگشت به حالت عادی (restore):
    هر برد در حجمِ کم، یک پله برمی‌گرداند: ۰.۲۵ → ۰.۵ → ۱.۰
    (پروتکل استاندارد بازیابی پراپ‌فرم‌ها؛ نه پرش ناگهانی به حجم کامل)

نکته: معامله‌ی سربه‌سر (r == 0) نه باخت حساب می‌شود نه برد — نردبان ثابت می‌ماند.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BreakerEvent:
    """رویداد قابل ثبت در ژورنال — ربات باید بتواند توضیح دهد چرا کاری کرد."""

    ts: datetime
    kind: str  # derate | deep_derate | diagnostic_queued | pause | halt | restore | resume | ack | monthly_halt
    detail: str
    streak: int
    size_multiplier: float


class CircuitBreaker:
    """موجودیت stateful — یک نمونه برای کل حساب، نه برای هر سیمبل.

    رشته باخت بین سیمبل‌ها مشترک است چون سقف ریسک ماهانه مشترک است
    (همان منطق قوانین پراپ).
    """

    def __init__(
        self,
        derate_at: int = 2,
        deep_derate_at: int = 3,
        pause_at: int = 4,
        halt_at: int = 6,
        pause_hours: float = 24.0,
        max_monthly_dd: float = 0.06,
    ) -> None:
        assert 1 <= derate_at <= deep_derate_at <= pause_at <= halt_at
        self.derate_at = derate_at
        self.deep_derate_at = deep_derate_at
        self.pause_at = pause_at
        self.halt_at = halt_at
        self.pause_hours = pause_hours
        self.max_monthly_dd = max_monthly_dd

        # سطح کاهنده: 0=عادی، 2=نصف، 3=ربع
        self._level = 0
        self._streak = 0
        self._paused_until: Optional[datetime] = None
        self._halted = False
        self._halt_reason = ""
        self.events: List[BreakerEvent] = []

    # ------------------------------------------------------------------ #
    # آپدیت وضعیت
    # ------------------------------------------------------------------ #
    def on_trade_closed(self, r_multiple: float, ts: datetime) -> List[BreakerEvent]:
        """بعد از بسته‌شدن هر معامله صدا زده می‌شود. r بر حسب مضرب R (مثبت=برد)."""
        out: List[BreakerEvent] = []

        if r_multiple < 0:  # ---- باخت
            self._streak += 1
            if self._streak >= self.halt_at:
                self._halted = True
                self._halt_reason = f"{self._streak} consecutive losses"
                out.append(self._ev(ts, "halt", f"FULL HALT: {self._halt_reason}; needs acknowledge()"))
            elif self._streak >= self.pause_at:
                self._paused_until = ts + timedelta(hours=self.pause_hours)
                out.append(
                    self._ev(ts, "pause", f"paused {self.pause_hours}h; post-mortem required")
                )
            elif self._streak >= self.deep_derate_at:
                self._level = 3
                out.append(self._ev(ts, "deep_derate", "size x0.25, A-grade setups only"))
                out.append(self._ev(ts, "diagnostic_queued", "regime/spread/news scan queued"))
            elif self._streak >= self.derate_at:
                self._level = max(self._level, 2)
                out.append(self._ev(ts, "derate", "size x0.50"))
                out.append(self._ev(ts, "diagnostic_queued", "regime/spread/news scan queued"))
        elif r_multiple > 0:  # ---- برد
            if self._level > 0:
                self._level -= 1
                out.append(self._ev(ts, "restore", f"win at reduced size -> x{self.size_multiplier()}"))
            self._streak = 0

        # r == 0 (سربه‌سر): عمداً هیچ — شواهد کافی برای هیچ تغییری نیست.

        self.events.extend(out)
        return out

    def on_monthly_drawdown(self, dd_fraction: float, ts: datetime) -> Optional[BreakerEvent]:
        """از سمت RiskEngine صدا زده می‌شود. dd_fraction منفی است (مثلاً -0.07)."""
        if dd_fraction <= -self.max_monthly_dd and not self._halted:
            self._halted = True
            self._halt_reason = f"monthly drawdown {dd_fraction:.1%} >= {self.max_monthly_dd:.0%}"
            ev = self._ev(ts, "monthly_halt", self._halt_reason)
            self.events.append(ev)
            return ev
        return None

    def acknowledge(self, ts: datetime) -> Optional[BreakerEvent]:
        """رفع توقف کامل — فقط بعد از بازاجرای walk-forward / بازبینی انسانی.

        سطح کاهنده عمداً حفظ می‌شود (حتی بعد از ack با حجم ۰.۲۵ شروع می‌کنیم).
        """
        if not self._halted:
            return None
        self._halted = False
        self._paused_until = None
        self._streak = 0
        ev = self._ev(ts, "ack", "halt cleared; still starting at reduced size")
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------ #
    # پرس‌وجو
    # ------------------------------------------------------------------ #
    def size_multiplier(self) -> float:
        return {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.25}[self._level]

    def allow_entry(self, grade: str, ts: datetime) -> Tuple[bool, str]:
        """گیت نهایی قبل از هر ورود. grade یکی از 'A' | 'B' | 'C'."""
        if self._halted:
            return False, f"halted: {self._halt_reason}"
        if self._paused_until is not None and ts < self._paused_until:
            return False, f"paused until {self._paused_until.isoformat()}"
        if self._level >= 3 and grade != "A":
            return False, "A-grade setups only during deep derate"
        return True, "ok"

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def streak(self) -> int:
        return self._streak

    def state(self) -> dict:
        """برای telemetry و ژورنال."""
        return {
            "level": self._level,
            "streak": self._streak,
            "size_multiplier": self.size_multiplier(),
            "paused_until": self._paused_until.isoformat() if self._paused_until else None,
            "halted": self._halted,
            "halt_reason": self._halt_reason or None,
        }

    # ------------------------------------------------------------------ #
    def _ev(self, ts: datetime, kind: str, detail: str) -> BreakerEvent:
        return BreakerEvent(
            ts=ts, kind=kind, detail=detail, streak=self._streak,
            size_multiplier=self.size_multiplier(),
        )
