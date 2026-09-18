"""قرارداد اجرا (Execution Adapter) — یک منطق، سه بستر اجرا.

    BacktestAdapter  → فاز ۱: موتور بک‌تست event-driven (همین‌جا، لینوکس)
    PaperAdapter     → فاز ۵: فوروارد دمو با دیتای لایو و اجرای شبیه‌سازی‌شده
    MT5Adapter       → فاز ۵: دموی واقعی MetaTrader 5 (روی ویندوز/VPS کاربر)

هیچ کد استراتژی/ریسکی نباید بداند زیرش کدام بستر است.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass(frozen=True)
class Order:
    symbol: str
    direction: int          # +1 / -1
    units: float
    order_type: str = "market"
    price: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    ts: Optional[datetime] = None
    meta: Dict = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Fill:
    order_id: str
    ts: datetime
    price: float
    units: float
    slippage: float = 0.0


class ExecutionAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, position_id: str, ts: datetime, price: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_positions(self) -> list:
        raise NotImplementedError
