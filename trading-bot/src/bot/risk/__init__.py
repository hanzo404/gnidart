"""برق‌گیر ضرر — تنها پیاده‌سازی ریسکِ زنده (v0.5.8: نسخهٔ دوم حذف شد).

تاریخچه: RiskEngine (engine.py) پیاده‌سازی موازی و مرده بود — لایو و
بک‌تست هر دو از CircuitBreaker + منطق خودِ runner/backtester استفاده
می‌کنند. «تک‌منبع حقیقت» (درس ممیزی ۴، P0-3): فقط همین می‌ماند.
"""
from .circuit_breaker import BreakerEvent, CircuitBreaker

__all__ = ["BreakerEvent", "CircuitBreaker"]
