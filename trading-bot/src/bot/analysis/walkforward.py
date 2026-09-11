"""هارنس Walk-Forward — لایهٔ صداقت (فاز ۲-ب).

مشکلی که حل می‌کند: هر پارامتری که با دیدن «کل داده» انتخاب شده،
نتیجه‌اش درون-نمونه است و قابل تعمیم نیست. این هارنس فرایند واقعیِ
تصمیم‌گیری را شبیه‌سازی می‌کند:

    برای هر ماهِ آزمون (OOS):
        ۱. فقط داده‌های «قبل از» آن ماه را ببین (پنجرهٔ IS، ۱۲ ماه)
        ۲. بهترین پارامترها را روی همان پنجره انتخاب کن
        ۳. با آن پارامترهای منجمد‌شده، آن یک ماه را معامله کن
        ۴. نتیجه را به مجموعهٔ OOS اضافه کن — دیگر هرگز دست نمی‌خورد

    تجمیع OOS = بهترین تخمین صادقانه از عملکرد آینده.

قواعد ضد-بیش‌برازش:
    - گرید کوچک (۲۷ ترکیب) — سطح پارامتریِ کوچک = ماشینِ کمتر برای拟合
    - حداقل ۳۰ معامله در IS؛ وگرنه پارامتر پیش‌فرض (بدون هیچ انتخابی)
    - رژیم‌ها روی کل تاریخ «علّی» محاسبه می‌شوند (گرم‌شدن اندیکاتور با
      دادهٔ واقعیِ قبل از پنجره، نه دادهٔ داخل پنجره)
    - مونت‌کارلو روی توالی معاملات OOS → توزیع واقعی افت سرمایه
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.regime.engine import RegimeConfig, RegimeEngine, session_open


@dataclass
class WFOConfig:
    is_months: int = 12
    oos_months: int = 1
    min_is_trades: int = 30
    adx_choices: tuple = (21.0, 23.0, 25.0)
    chaos_choices: tuple = (1.7, 1.8, 1.9)
    rr_choices: tuple = (2.0, 2.5, 3.0)
    defaults: tuple = (23.0, 1.8, 2.5)     # (adx, chaos, rr) — بدون بهینه‌سازی
    start_equity: float = 3000.0
    fixed_lots: float = 0.01
    slippage_usd: float = 0.05
    spread_gate_usd: float = 0.40
    mc_paths: int = 2000
    mc_seed: int = 42


# ------------------------------------------------------------------ #
# قطعات خالص (تست‌پذیر بدون داده)
# ------------------------------------------------------------------ #
def monthly_windows(month_keys: np.ndarray, is_months: int,
                    oos_months: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """ماه‌ها (آرایهٔ 'YYYY-MM') → ماسک‌های (IS, OOS) بدون هیچ هم‌پوشانی.

    تضمین: max(زمان IS) < min(زمان OOS) — تست ضد-نشتی همین را قفل می‌کند.
    """
    months = sorted(set(month_keys))
    wins = []
    i = is_months
    while i < len(months):
        is_set = set(months[i - is_months:i])
        oos_set = set(months[i:i + oos_months])
        if not oos_set:
            break
        is_mask = np.isin(month_keys, list(is_set))
        oos_mask = np.isin(month_keys, list(oos_set))
        wins.append((is_mask, oos_mask))
        i += oos_months
    return wins


def select_best(scores: list[tuple[tuple, float, int]],
                defaults: tuple, min_trades: int) -> tuple:
    """scores: [(params, pnl, n_trades), ...] → پارامتر برنده.

    کمتر از حداقل معامله = نمرهٔ −بی‌نهایت؛ اگر هیچ‌کدام نگذرند → پیش‌فرض.
    """
    best, best_pnl = defaults, -np.inf
    for params, pnl, n in scores:
        if n >= min_trades and pnl > best_pnl:
            best, best_pnl = params, pnl
    return best


def bootstrap_maxdd(pnls: np.ndarray, paths: int = 2000, seed: int = 42,
                    start_equity: float = 3000.0) -> dict:
    """مونت‌کارلو روی توالی معاملات (نمونه‌گیری با جای‌گزینی).

    جواب به «اگر ترتیب همان معاملات جور دیگری می‌بود، افت سرمایه چه بود؟»
    — چون در واقعیت ترتیبِ خوب/بد شانسی است.
    """
    rng = np.random.default_rng(seed)
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {}
    draws = rng.choice(pnls, size=(paths, len(pnls)), replace=True)
    eq = start_equity + np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq - peak) / peak
    maxdd = dd.min(axis=1)                     # هر مسیر: بدترین افت
    terminal = eq[:, -1]
    return {
        "maxdd_p5": float(np.percentile(maxdd, 5)),
        "maxdd_p50": float(np.percentile(maxdd, 50)),
        "maxdd_p95": float(np.percentile(maxdd, 95)),
        "terminal_p5": float(np.percentile(terminal, 5)),
        "terminal_p50": float(np.percentile(terminal, 50)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "p_dd_below_15": float((maxdd < -0.15).mean()),
        "p_dd_below_25": float((maxdd < -0.25).mean()),
    }


# ------------------------------------------------------------------ #
# هارنس اصلی
# ------------------------------------------------------------------ #
@dataclass
class WFOResult:
    oos_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    chosen_freq: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    static_metrics: dict = field(default_factory=dict)
    mc: dict = field(default_factory=dict)


class WalkForwardOptimizer:
    def __init__(self, bars: pd.DataFrame, h4: pd.DataFrame,
                 cfg: WFOConfig | None = None):
        self.cfg = cfg or WFOConfig()
        self.bars = bars.sort_values("time").reset_index(drop=True)
        self.h4 = h4
        self._regimes: dict[tuple, np.ndarray] = {}

    # ---------------- رژیم‌ها (یک‌بار در ازای هر ترکیب) ---------------- #
    def regime_for(self, adx: float, chaos: float) -> np.ndarray:
        key = (adx, chaos)
        if key not in self._regimes:
            rc = RegimeConfig(adx_enter=adx, adx_exit=adx - 3.0,
                              chaos_ratio=chaos)
            self._regimes[key] = RegimeEngine(rc).compute(
                self.bars)["regime"].to_numpy()
        return self._regimes[key]

    # ---------------- اجرای یک بک‌تست گیت‌شده ---------------- #
    def _run(self, mask: np.ndarray, adx: float, chaos: float,
             rr: float) -> tuple[float, int, pd.DataFrame]:
        cfg = self.cfg
        bars = self.bars[mask].reset_index(drop=True)
        regime = self.regime_for(adx, chaos)[mask]
        sess = session_open(bars["time"])
        strat = GatedStrategy(V0Strategy(self.h4, rr=rr), regime, sess)
        bt_cfg = BacktestConfig(
            start_equity=cfg.start_equity, fixed_lots=cfg.fixed_lots,
            max_positions=1, cooldown_bars=1, slippage_usd=cfg.slippage_usd,
            spread_gate_usd=cfg.spread_gate_usd)
        res = Backtester(bars, strat, bt_cfg).run()
        n = len(res.trades)
        pnl = float(res.trades["pnl"].sum()) if n else 0.0
        return pnl, n, res.trades

    # ---------------- حلقهٔ اصلی ---------------- #
    def run(self) -> WFOResult:
        cfg = self.cfg
        keys = self.bars["time"].dt.strftime("%Y-%m").to_numpy()
        windows = monthly_windows(keys, cfg.is_months, cfg.oos_months)
        grid = list(product(cfg.adx_choices, cfg.chaos_choices, cfg.rr_choices))

        oos_frames, rows, chosen = [], [], []
        static_frames = []
        for is_mask, oos_mask in windows:
            # ۱) انتخاب پارامتر فقط با IS
            scores = []
            for adx, chaos, rr in grid:
                pnl, n, _ = self._run(is_mask, adx, chaos, rr)
                scores.append(((adx, chaos, rr), pnl, n))
            best = select_best(scores, cfg.defaults, cfg.min_is_trades)
            chosen.append(best)

            # ۲) OOS با پارامتر منجمد
            pnl, n, trades = self._run(oos_mask, *best)
            oos_frames.append(trades)
            # ۳) پایهٔ مقایسه: پیش‌فرض ثابت روی همان OOS
            spnl, sn, strades = self._run(oos_mask, *cfg.defaults)
            static_frames.append(strades)

            mkey = ",".join(sorted(set(keys[oos_mask])))
            rows.append({"oos_month": mkey, "params": best,
                         "trades": n, "pnl": pnl,
                         "static_pnl": spnl, "static_trades": sn})

        oos = pd.concat(oos_frames, ignore_index=True) if oos_frames \
            else pd.DataFrame()
        static = pd.concat(static_frames, ignore_index=True) \
            if static_frames else pd.DataFrame()

        result = WFOResult(
            oos_trades=oos,
            per_window=pd.DataFrame(rows),
            chosen_freq=pd.Series(map(str, chosen)).value_counts()
            .rename_axis("params").reset_index(name="count"),
        )
        result.metrics = self._metrics(oos)
        result.static_metrics = self._metrics(static)
        result.mc = bootstrap_maxdd(oos["pnl"].to_numpy()
                                    if len(oos) else np.array([]),
                                    cfg.mc_paths, cfg.mc_seed,
                                    cfg.start_equity)
        return result

    @staticmethod
    def _metrics(trades: pd.DataFrame) -> dict:
        if trades is None or trades.empty:
            return {"n_trades": 0}
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        gross_w = wins["pnl"].sum()
        gross_l = -losses["pnl"].sum()
        eq = 3000.0 + trades["pnl"].cumsum()
        peak = eq.cummax()
        dd = ((eq - peak) / peak).min()
        return {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "profit_factor": (gross_w / gross_l) if gross_l > 0 else np.inf,
            "total_pnl": trades["pnl"].sum(),
            "expectancy_r": trades["r"].mean(),
            "trade_level_maxdd_pct": dd,
        }
