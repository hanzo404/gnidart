"""قرارداد استراتژی — هر استراتژی باید این اینترفیس را پیاده کند.

قواعد غیرقابل‌مذاکره (ضد overfitting / ضد نشتی داده):
    1.因果性: سیگنال فقط از کندل‌های «بسته‌شده» تا لحظه t؛ هرگز از آینده.
    2. هر Signal باید stop داشته باشد — معامله بدون استاپ = باگ، نه معامله.
    3. grade کیفیت ستاپ است (A/B/C) — برق‌گیر ضرر در حالت عمیق فقط A می‌پذیرد.
    4. features: دیکشنری فیچرهای لحظه ورود برای ژورنال (بستر meta-labeling فاز ۶).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Signal:
    ts: datetime
    symbol: str
    direction: int          # +1 خرید / -1 فروش
    entry: float
    stop: float
    target: Optional[float]
    grade: str              # 'A' | 'B' | 'C'
    strategy: str
    reason: str
    features: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """کلاس پایه. هر استراتژی فقط در رژیم مجاز خودش فعال می‌شود (فاز ۲)."""

    name: str = "base"
    timeframe: str = "M15"
    allowed_regimes: tuple = ()   # خالی = همه (فعلاً)

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        """df: OHLCV کندل‌های بسته‌شده تا حالا. خروجی: سیگنال‌های جدید."""
        raise NotImplementedError
