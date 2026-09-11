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
    # هماهنگ با config.yaml — نردبان تنظیمی فاز ۳ (شواهد G4)
    derate_at: int = 3
    deep_derate_at: int = 4
    pause_at: int = 5
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
class LiveConfig:
    """فاز ۵ — اجرای فوروارد دمو."""
    direction_filter: str = "long"   # شواهد فاز ۴: فروش‌ها خالص‌زیان‌ده بودند
    utc_offset: str = "auto"         # auto = قاعدهٔ EET/EEST (سرور MetaQuotes)
    history_bars: int = 800          # گرم‌شدن ADX/EMA/ATR
    poll_seconds: int = 20
    magic: int = 954001              # شناسهٔ سفارش‌های ربات در MT5


@dataclass
class SymbolProfile:
    """پارامترهای وابسته به نماد — فاز ۷ (چند-نمادی).

    هر چیزی که «مقیاس دلاریِ» نماد است این‌جاست؛ بقیهٔ ربات
    نماد-ناوابسته می‌ماند. اعداد = مستقیم از شواهد بک‌تست همان نماد:
      XAUUSD: فاز ۳-۶ (walk-forward طلا)
      XAGUSD: فاز ۷ first-look (k = نسبت ATR نقره/طلا = 0.02186)
    """
    contract_size: float = 100.0    # دلار به ازای ۱ واحد حرکت قیمت × ۱ لات (طلا ۱۰۰اونس، نقره ۵۰۰۰اونس)
    sl_pad: float = 0.50            # بافر استاپ دلاری (≈ نصف ATR14 میانهٔ M15 همان نماد)
    max_spread_usd: float = 0.40    # گیت اسپرد
    max_lots: float = 0.10          # سقف عقل‌سنجی حجم دمو
    min_lot_policy: str = "force"   # force = حداقل‌لات اجباری (رفتار طلا) | skip = رد اگر حجم < حداقل
    digits: int = 2                 # ارقام اعشار قیمت (نمایش/لاگ)
    baseline_wr: float = 0.348      # پایهٔ نرخ‌برد برای تشخیص/پست‌مورتم همان نماد
    direction: str = None           # فیلتر جهت این آستین؛ None → از live.direction_filter


# پیش‌فرض‌های مستند — config.yaml می‌تواند هر کدام را override کند
DEFAULT_PROFILES: Dict[str, "SymbolProfile"] = {
    "XAUUSD": SymbolProfile(),  # = رفتار تک‌نمادی فاز ۵، دست‌نخورده
    "XAGUSD": SymbolProfile(
        contract_size=5000.0, sl_pad=0.01093, max_spread_usd=0.06,
        max_lots=0.05, min_lot_policy="skip", digits=3, baseline_wr=0.308),
}


@dataclass
class BotConfig:
    project: str = "gnidart-trading-bot"
    mode: str = "backtest"
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    symbol_profiles: Dict[str, SymbolProfile] = field(
        default_factory=lambda: {k: SymbolProfile(**vars(v))
                                 for k, v in DEFAULT_PROFILES.items()})

    def profile_for(self, symbol: str) -> SymbolProfile:
        """پروفایل نماد؛ نماد ناشناخته → پروفایل طلا + هشدار در لاگ اجرا."""
        return self.symbol_profiles.get(symbol, SymbolProfile())

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
            ("live", cfg.live),
        ):
            if isinstance(raw.get(section), dict):
                for k, v in raw[section].items():
                    if hasattr(target, k):
                        setattr(target, k, v)
        # پروفایل‌های نماد (فاز ۷) — روی پیش‌فرض‌های مستند override می‌شوند
        if isinstance(raw.get("symbol_profiles"), dict):
            for sym, d in raw["symbol_profiles"].items():
                base = dict(vars(cfg.symbol_profiles.get(sym, SymbolProfile())))
                base.update({k: v for k, v in d.items() if k in base})
                cfg.symbol_profiles[sym] = SymbolProfile(**base)
        # sessions: list -> tuple
        cfg.trading.sessions_utc = {
            k: tuple(v) for k, v in cfg.trading.sessions_utc.items()
        }
        return cfg
