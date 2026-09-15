"""دریافت داده از ترمینال MetaTrader 5 — فقط ویندوز.

نقش در معماری: منبع داده فوروارد دمو (فاز ۵) + اعتبارسنجی داده.
بک‌تست اصلی روی داده Dukascopy انجام می‌شود (عمق و کیفیت بالاتر).

نکته مهم ضد نشتی داده: آخرین کندلِ برگشتی از MT5 «در حال تشکیل» است؛
به‌صورت پیش‌فرض حذف می‌شود تا استراتژی‌ها فقط کندلِ بسته‌شده ببینند
(قرارداد strategy/base.py، قاعده ۱).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:  # پکیج MetaTrader5 فقط روی ویندوز نصب می‌شود
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:  # pragma: no cover - روی لینوکس/سندباکس
    mt5 = None
    _HAS_MT5 = False

_TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class MT5DataProvider:
    """اتصال به یک ترمینال MT5 (باز و لاگین‌شده) برای کندل و اطلاعات اکانت."""

    def __init__(self, symbol: str = "EURUSD", terminal_path: Optional[str] = None) -> None:
        if not _HAS_MT5:
            raise RuntimeError(
                "پکیج MetaTrader5 نصب نیست — روی ویندوز: py -m pip install MetaTrader5"
            )
        self.symbol = symbol
        self.terminal_path = terminal_path
        self._connected = False

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        kwargs = {"path": self.terminal_path} if self.terminal_path else {}
        if not mt5.initialize(**kwargs):
            code, desc = mt5.last_error()
            raise ConnectionError(
                f"اتصال به ترمینال MT5 ناموفق ({code}): {desc}\n"
                "چک‌لیست: ترمینال باز است؟ لاگین شده؟ "
                "(چند ترمینال داری → terminal_path بده)"
            )
        self._connected = True

    def close(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5DataProvider":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def candles(self, timeframe: str = "M15", count: int = 1000,
                closed_only: bool = True) -> pd.DataFrame:
        """کندل‌های OHLCV: time, open, high, low, close, tick_volume, spread.

        closed_only=True → کندل در حال تشکیل (آخرین ردیف) حذف می‌شود.
        """
        if not self._connected:
            self.connect()
        tf_name = _TIMEFRAMES.get(timeframe.upper())
        if tf_name is None:
            raise ValueError(f"تایم‌فریم پشتیبانی نمی‌شود: {timeframe}")
        tf = getattr(mt5, tf_name)

        if mt5.symbol_info(self.symbol) is None:
            raise ValueError(
                f"سیمبل '{self.symbol}' در این بروکر پیدا نشد — Market Watch را چک کن؛ "
                "بعضی بروکرها پسوند دارند (مثلاً EURUSD.r یا EURUSD.a)"
            )
        mt5.symbol_select(self.symbol, True)

        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            code, desc = mt5.last_error()
            raise RuntimeError(f"دریافت کندل ناموفق ({code}): {desc}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        if closed_only:
            df = df.iloc[:-1].copy()
        return df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]

    def account_summary(self) -> dict:
        if not self._connected:
            self.connect()
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("account_info ناموفق — لاگین ترمینال چک شود")
        return {
            "login": acc.login,
            "server": acc.server,
            "balance": acc.balance,
            "currency": acc.currency,
            "is_demo": acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO,
            "leverage": acc.leverage,
        }

    def live_quote(self) -> dict:
        if not self._connected:
            self.connect()
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise ValueError(f"سیمبل '{self.symbol}' پیدا نشد")
        return {
            "bid": info.bid,
            "ask": info.ask,
            "point": info.point,
            "spread_points": (info.ask - info.bid) / info.point,
            "tick_epoch": float(getattr(info, "time", 0) or 0),
        }

    def symbol_spec(self) -> dict:
        """مشخصات واقعی نماد از بروکر — برای اعتبارسنجی پروفایل (ممیزی ۳)."""
        if not self._connected:
            self.connect()
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return {}
        return {
            "trade_contract_size": getattr(info, "trade_contract_size", None),
            "digits": getattr(info, "digits", None),
            "volume_min": getattr(info, "volume_min", None),
            "volume_step": getattr(info, "volume_step", None),
            "volume_max": getattr(info, "volume_max", None),
            "point": getattr(info, "point", None),
        }
