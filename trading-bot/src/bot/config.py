"""کانفیگ پروژه — تمام پارامترها از YAML؛ بدون عدد جادویی در کد."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    decision_timeframe: str = "M15"
    trend_timeframe: str = "H1"
    entry_timeframe: str = "M5"
    trade_sessions: List[str] = field(default_factory=lambda: ["london", "newyork"])
    sessions_utc: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "asia": (0, 8),
            "london": (7, 16),
            "newyork": (12, 21),
        }
    )
    max_trades_per_day: int = 3
    max_open_positions: int = 1
    max_spread_usd: float = 0.40


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.03
    max_monthly_loss: float = 0.06


@dataclass
class BreakerConfig:
    derate_at: int = 2
    deep_derate_at: int = 3
    pause_at: int = 4
    halt_at: int = 6
    pause_hours: float = 24.0


@dataclass
class DataConfig:
    source: str = "dukascopy"
    cache_dir: str = "data/cache"


@dataclass
class JournalConfig:
    path: str = "data/journal.db"


@dataclass
class BotConfig:
    project: str = "gnidart-trading-bot"
    mode: str = "backtest"
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)

    @staticmethod
    def load(path: str | pathlib.Path) -> "BotConfig":
        p = pathlib.Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
        cfg = BotConfig()
        if not raw:
            return cfg
        cfg.project = raw.get("project", cfg.project)
        cfg.mode = raw.get("mode", cfg.mode)
        for section, target in (
            ("trading", cfg.trading),
            ("risk", cfg.risk),
            ("breaker", cfg.breaker),
            ("data", cfg.data),
            ("journal", cfg.journal),
        ):
            if isinstance(raw.get(section), dict):
                for k, v in raw[section].items():
                    if hasattr(target, k):
                        setattr(target, k, v)
        # sessions: list -> tuple
        cfg.trading.sessions_utc = {
            k: tuple(v) for k, v in cfg.trading.sessions_utc.items()
        }
        return cfg
