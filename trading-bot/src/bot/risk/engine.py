"""موتور ریسک — سایز پوزیشن، محافظ دراودان روزانه/ماهانه.

اصول (از کتاب‌های الدر):
    - ریسک هر معامله = درصد ثابتی از سرمایه × ضریب برق‌گیر
    - ۳٪ روزانه → توقف تا فردا
    - ۶٪ ماهانه → توقف کامل (برق‌گیر halt می‌کند)

فرمول سایز (تقریب حساب کوت/USD برای شروع؛ MT5 adapter در فاز ۵ دقیقش می‌کند):
    units = risk_amount / |entry - stop|
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from .circuit_breaker import CircuitBreaker


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    units: float
    risk_amount: float
    size_multiplier: float
    reason: str


class DailyLossGuard:
    """اگر ضرر روزانه از حد گذشت، تا فردا ورود ممنوع."""

    def __init__(self, max_daily_loss: float = 0.03) -> None:
        self.max_daily_loss = max_daily_loss
        self._day: Optional[str] = None
        self._day_peak: float = 0.0

    def update(self, equity: float, ts: datetime) -> None:
        day = ts.strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self._day_peak = day, equity
        self._day_peak = max(self._day_peak, equity)

    def blocked(self, equity: float, ts: datetime) -> Tuple[bool, str]:
        day = ts.strftime("%Y-%m-%d")
        if day != self._day:
            return False, ""
        dd = (equity - self._day_peak) / self._day_peak
        if dd <= -self.max_daily_loss:
            return True, f"daily drawdown {dd:.1%} <= -{self.max_daily_loss:.0%}"
        return False, ""


class MonthlyDrawdownGuard:
    """۶٪ ماهانه (قانون الدر) → توقف کامل تا acknowledge."""

    def __init__(self, max_monthly_loss: float = 0.06) -> None:
        self.max_monthly_loss = max_monthly_loss
        self._month: Optional[str] = None
        self._month_peak: float = 0.0

    def update(self, equity: float, ts: datetime, breaker: CircuitBreaker) -> Optional[float]:
        """هر تغییر equity صدا زده شود؛ اگر از حد گذشت، breaker را halt می‌کند."""
        month = ts.strftime("%Y-%m")
        if month != self._month:
            self._month, self._month_peak = month, equity
        self._month_peak = max(self._month_peak, equity)
        dd = (equity - self._month_peak) / self._month_peak
        if dd <= -self.max_monthly_loss:
            breaker.on_monthly_drawdown(dd, ts)
        return dd


class RiskEngine:
    def __init__(
        self,
        breaker: CircuitBreaker,
        risk_per_trade: float = 0.005,
        max_daily_loss: float = 0.03,
        max_monthly_loss: float = 0.06,
    ) -> None:
        self.breaker = breaker
        self.risk_per_trade = risk_per_trade
        self.daily = DailyLossGuard(max_daily_loss)
        self.monthly = MonthlyDrawdownGuard(max_monthly_loss)

    def on_equity_update(self, equity: float, ts: datetime) -> None:
        self.daily.update(equity, ts)
        self.monthly.update(equity, ts, self.breaker)

    def size_for(
        self,
        equity: float,
        entry: float,
        stop: float,
        grade: str,
        ts: datetime,
    ) -> RiskDecision:
        """گیت + سایز. سه لایه رد: halt/pause برق‌گیر، A-grade-only، حد روزانه."""
        ok, reason = self.breaker.allow_entry(grade, ts)
        if not ok:
            return RiskDecision(False, 0.0, 0.0, self.breaker.size_multiplier(), reason)

        blocked, why = self.daily.blocked(equity, ts)
        if blocked:
            return RiskDecision(False, 0.0, 0.0, 0.0, why)

        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return RiskDecision(False, 0.0, 0.0, self.breaker.size_multiplier(), "stop == entry")

        mult = self.breaker.size_multiplier()
        risk_amount = equity * self.risk_per_trade * mult
        units = risk_amount / stop_distance
        return RiskDecision(True, units, risk_amount, mult, "ok")
