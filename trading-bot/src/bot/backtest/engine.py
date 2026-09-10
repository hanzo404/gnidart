"""موتور بک‌تست رویداد-محور — فاز ۱.

اصل طراحی: «هیچ چیزی که در واقعیت هزینه دارد، اینجا رایگان نیست.»

واقع‌گرایی‌های پیاده‌سازی‌شده:
    - سیگنال در close کندل i صادر می‌شود؛ پر شدن در open کندل i+1
      (ضد-نشتی: هیچ تصمیمی با دادهٔ آینده گرفته نمی‌شود)
    - کندل‌ها BID هستند:
        * ورود BUY  با ask  = open + spread + slippage
        * ورود SELL با bid  = open − slippage
        * خروج BUY  (فروش) با bid
        * خروج SELL (خرید پوشش) با ask = قیمت + spread + slippage
      → اسپرد دقیقاً یک‌بار در هر معامله از سمت ask وصل می‌شود، مثل واقعیت.
    - گپ آخر هفته/اخبار: اگر کندلِ بعد از گپ بیرون از استاپ باز شود،
      پر شدن با «قیمت open» است نه قیمت استاپ (ضررِ گپ واقعی).
    - اگر در یک کندل هم استاپ و هم تارگت لمس شوند → استاپ اول
      (فرض محافظه‌کارانه؛ هیچ‌کس نمی‌داند کدام اول بوده).
    - TP بر اساس R از «قیمت واقعی پر شدن» محاسبه می‌شود، نه سیگنال.
    - halt روی نابودی حساب (equity ≤ 0): مثل margin call بروکر.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Order:
    """سیگنال ورود؛ در open کندل بعدی پر می‌شود."""
    direction: int            # +1 خرید / -1 فروش
    stop: float               # حد ضرر (قیمت BID)
    rr: float = 2.5           # TP = entry ± rr × ریسک
    tag: str = ""


@dataclass
class BacktestConfig:
    start_equity: float = 3000.0
    # سایزینگ: اگر risk_pct تنظیم شود ریسک درصدی، وگرنه لات ثابت
    risk_pct: Optional[float] = None
    fixed_lots: float = 0.01
    contract_oz: float = 100.0        # XAUUSD: هر لات = ۱۰۰ اونس
    min_lots: float = 0.01
    lot_step: float = 0.01
    slippage_usd: float = 0.05        # به ازای هر اونس، هر سمت
    spread_gate_usd: Optional[float] = None   # ورود ممنوع اگر اسپرد بیشتر باشد
    max_positions: int = 1
    cooldown_bars: int = 0            # حداقل فاصله بین دو ورود (تعداد کندل)
    stop_first_same_bar: bool = True
    halt_equity: float = 0.0          # توقف مثل margin call
    # ریسک‌فری خودکار: وقتی close به be_at_frac مسیرِ TP رسید، استاپ به
    # نقطه‌ی ورود منتقل می‌شود (None = خاموش — پیش‌فرض، رفتار تغییر نمی‌کند)
    be_at_frac: Optional[float] = None


class Strategy(Protocol):
    def prepare(self, bars: pd.DataFrame) -> None: ...
    def on_bar(self, i: int) -> Optional[Order]: ...


@dataclass
class _Pos:
    direction: int
    entry_i: int
    entry_fill: float
    stop: float
    target: float
    lots: float
    risk_usd: float
    tag: str
    be_done: bool = False


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame              # time, equity (شامل سود شناوری)
    metrics: dict = field(default_factory=dict)
    halted: bool = False
    halt_time: Optional[pd.Timestamp] = None


class Backtester:
    def __init__(self, bars: pd.DataFrame, strategy: Strategy,
                 config: BacktestConfig | None = None, risk_policy=None):
        """risk_policy: شیء با قرارداد RiskPolicy (ببین backtest/risk.py)
        — گیت ورود + ضریب سایز + رویدادهای معامله/کندل."""
        if bars.empty:
            raise ValueError("دادهٔ کندل خالی است")
        self.cfg = config or BacktestConfig()
        self.bars = bars.sort_values("time").reset_index(drop=True)
        self.strategy = strategy
        self.policy = risk_policy
        self._min_lot_clamps = 0

    # ------------------------------------------------------------------ #
    def _size_lots(self, stop_dist: float, equity: float,
                   mult: float = 1.0) -> float:
        if self.cfg.risk_pct is None:
            lots = np.floor(self.cfg.fixed_lots * mult / self.cfg.lot_step) \
                * self.cfg.lot_step
        else:
            if stop_dist <= 0:
                return 0.0
            raw = (equity * self.cfg.risk_pct * mult) / \
                (stop_dist * self.cfg.contract_oz)
            lots = np.floor(raw / self.cfg.lot_step) * self.cfg.lot_step
        if lots < self.cfg.min_lots:
            # کفِ لات بروکر — صادقانه بشمار تا گزارش بگوید
            if equity > 0 and mult < 1.0:
                self._min_lot_clamps += 1
            return self.cfg.min_lots
        return lots

    def _exit_check(self, p: _Pos, i: int) -> tuple[Optional[float], Optional[str]]:
        o, h, l = self._o[i], self._h[i], self._l[i]
        sp, slip = self._sp[i], self.cfg.slippage_usd
        if p.direction > 0:                                   # خرید
            if o <= p.stop:
                return o - slip, "gap_stop"
            if o >= p.target:
                return o - slip, "gap_target"
            hit_stop, hit_tg = l <= p.stop, h >= p.target
            if hit_stop and self.cfg.stop_first_same_bar:
                return p.stop - slip, "stop"
            if hit_tg:
                return p.target - slip, "target"
            if hit_stop:
                return p.stop - slip, "stop"
        else:                                                  # فروش
            if o >= p.stop:
                return o + sp + slip, "gap_stop"
            if o <= p.target:
                return o + sp + slip, "gap_target"
            hit_stop, hit_tg = h >= p.stop, l <= p.target
            if hit_stop and self.cfg.stop_first_same_bar:
                return p.stop + sp + slip, "stop"
            if hit_tg:
                return p.target + sp + slip, "target"
            if hit_stop:
                return p.stop + sp + slip, "stop"
        return None, None

    # ------------------------------------------------------------------ #
    def run(self) -> BacktestResult:
        cfg = self.cfg
        self._o = self.bars["open"].to_numpy(float)
        self._h = self.bars["high"].to_numpy(float)
        self._l = self.bars["low"].to_numpy(float)
        self._c = self.bars["close"].to_numpy(float)
        self._sp = (self.bars["spread_usd"].to_numpy(float)
                    if "spread_usd" in self.bars.columns
                    else np.zeros(len(self.bars)))
        t = self.bars["time"].to_numpy()
        n = len(self.bars)
        self.strategy.prepare(self.bars)

        positions: list[_Pos] = []
        trades: list[dict] = []
        pending: Optional[Order] = None
        last_entry_i = -10**9
        realized = cfg.start_equity
        equity_curve = np.full(n, np.nan)
        halted, halt_time = False, None

        for i in range(n):
            # --- ۱) پر کردن سفارش معوق در open این کندل ------------------
            if pending is not None:
                allowed = (len(positions) < cfg.max_positions
                           and (i - last_entry_i) >= cfg.cooldown_bars
                           and (cfg.spread_gate_usd is None
                                or self._sp[i] <= cfg.spread_gate_usd))
                if allowed and self.policy is not None:
                    allowed = self.policy.allow_entry(t[i])
                if allowed:
                    d, stop, rr = pending.direction, pending.stop, pending.rr
                    if d > 0:
                        fill = self._o[i] + self._sp[i] + cfg.slippage_usd
                    else:
                        fill = self._o[i] - cfg.slippage_usd
                    dist = abs(fill - stop)
                    mult = (self.policy.size_multiplier()
                            if self.policy is not None else 1.0)
                    lots = self._size_lots(dist, realized, mult)
                    if lots > 0 and dist > 0:
                        target = (fill + rr * (fill - stop) if d > 0
                                  else fill - rr * (stop - fill))
                        positions.append(_Pos(d, i, fill, stop, target, lots,
                                              dist * lots * cfg.contract_oz,
                                              pending.tag))
                        last_entry_i = i
                pending = None  # سیگنال فقط یک کندل اعتبار دارد

            # --- ۲) مدیریت خروج روی این کندل -----------------------------
            still: list[_Pos] = []
            for p in positions:
                px, reason = self._exit_check(p, i)
                if px is None:
                    still.append(p)
                    continue
                if p.direction > 0:
                    pnl = (px - p.entry_fill) * p.lots * cfg.contract_oz
                else:
                    pnl = (p.entry_fill - px) * p.lots * cfg.contract_oz
                realized += pnl
                trades.append({
                    "entry_time": t[p.entry_i], "exit_time": t[i],
                    "direction": p.direction, "entry": p.entry_fill,
                    "exit": px, "stop": p.stop, "target": p.target,
                    "lots": p.lots, "pnl": pnl, "r": pnl / p.risk_usd if p.risk_usd else 0.0,
                    "reason": reason, "bars_held": i - p.entry_i, "tag": p.tag,
                })
                if self.policy is not None:
                    self.policy.on_trade_closed(trades[-1]["r"], t[i], realized)
            positions = still

            # --- ۲.۵) ریسک‌فری خودکار (close این کندل → استاپ کندل بعد) ---
            if cfg.be_at_frac is not None:
                for p in positions:
                    if p.be_done:
                        continue
                    if p.direction > 0:
                        trig = p.entry_fill + (p.target - p.entry_fill) \
                            * cfg.be_at_frac
                        if self._c[i] >= trig:
                            p.stop = max(p.stop, p.entry_fill)
                            p.be_done = True
                    else:
                        trig = p.entry_fill - (p.entry_fill - p.target) \
                            * cfg.be_at_frac
                        if self._c[i] <= trig:
                            p.stop = min(p.stop, p.entry_fill)
                            p.be_done = True

            # --- ۳) equity شناوری + چک نابودی ----------------------------
            float_pnl = 0.0
            for p in positions:
                if p.direction > 0:
                    float_pnl += (self._c[i] - p.entry_fill) * p.lots * cfg.contract_oz
                else:
                    float_pnl += (p.entry_fill - self._c[i]) * p.lots * cfg.contract_oz
            equity_curve[i] = realized + float_pnl
            if self.policy is not None:
                self.policy.on_bar_close(t[i], float(equity_curve[i]))
            if equity_curve[i] <= cfg.halt_equity:
                halted, halt_time = True, pd.Timestamp(t[i])
                for p in positions:  # بروکر همه را می‌بندد
                    pnl = (self._c[i] - p.entry_fill) * p.lots * cfg.contract_oz \
                        if p.direction > 0 else \
                        (p.entry_fill - self._c[i]) * p.lots * cfg.contract_oz
                    realized += pnl
                    trades.append({
                        "entry_time": t[p.entry_i], "exit_time": t[i],
                        "direction": p.direction, "entry": p.entry_fill,
                        "exit": float(self._c[i]), "stop": p.stop,
                        "target": p.target, "lots": p.lots, "pnl": pnl,
                        "r": pnl / p.risk_usd if p.risk_usd else 0.0,
                        "reason": "margin_call", "bars_held": i - p.entry_i,
                        "tag": p.tag,
                    })
                equity_curve[i:] = realized
                break

            # --- ۴) سیگنال در close این کندل ------------------------------
            sig = self.strategy.on_bar(i)
            if sig is not None:
                pending = sig

        else:
            # پایان داده: پوزیشن‌های باز با close آخرین کندل بسته می‌شوند
            last = n - 1
            for p in positions:
                pnl = (self._c[last] - p.entry_fill) * p.lots * cfg.contract_oz \
                    if p.direction > 0 else \
                    (p.entry_fill - self._c[last]) * p.lots * cfg.contract_oz
                realized += pnl
                trades.append({
                    "entry_time": t[p.entry_i], "exit_time": t[last],
                    "direction": p.direction, "entry": p.entry_fill,
                    "exit": float(self._c[last]), "stop": p.stop,
                    "target": p.target, "lots": p.lots, "pnl": pnl,
                    "r": pnl / p.risk_usd if p.risk_usd else 0.0,
                    "reason": "end_of_data", "bars_held": last - p.entry_i,
                    "tag": p.tag,
                })
                equity_curve[last] = realized

        trades_df = pd.DataFrame(trades)
        eq = pd.DataFrame({"time": t, "equity": equity_curve})
        res = BacktestResult(trades_df, eq, halted=halted, halt_time=halt_time)
        res.metrics = self._metrics(trades_df, eq, cfg)
        res.metrics["min_lot_clamps"] = self._min_lot_clamps
        return res

    # ------------------------------------------------------------------ #
    @staticmethod
    def _metrics(trades: pd.DataFrame, eq: pd.DataFrame,
                 cfg: BacktestConfig) -> dict:
        m: dict = {"start_equity": cfg.start_equity}
        if trades.empty:
            return m | {"n_trades": 0}
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        gross_w = wins["pnl"].sum()
        gross_l = -losses["pnl"].sum()
        eq_peak = eq["equity"].cummax()
        dd = eq["equity"] - eq_peak
        m.update({
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "profit_factor": (gross_w / gross_l) if gross_l > 0 else np.inf,
            "total_pnl": trades["pnl"].sum(),
            "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
            "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
            "expectancy": trades["pnl"].mean(),
            "expectancy_r": trades["r"].mean(),
            "max_dd": dd.min(),
            "max_dd_pct": (dd / eq_peak).min(),
            "final_equity": eq["equity"].iloc[-1],
            "avg_bars_held": trades["bars_held"].mean(),
        })
        return m


def print_metrics(title: str, res: BacktestResult) -> None:
    m = res.metrics
    print(f"\n── {title} ─────────────────────────────")
    if m.get("n_trades", 0) == 0:
        print("  هیچ معامله‌ای رخ نداد.")
        return
    print(f"  تعداد معاملات: {m['n_trades']:,}")
    print(f"  نرخ برد: {m['win_rate']:.1%} | PF: {m['profit_factor']:.2f}")
    print(f"  سود/زیان کل: ${m['total_pnl']:,.0f} | equity نهایی: ${m['final_equity']:,.0f}")
    print(f"  انتظار ریاضی: ${m['expectancy']:.2f}/معامله | {m['expectancy_r']:.3f}R")
    print(f"  میانگین برد ${m['avg_win']:.2f} / میانگین باخت ${m['avg_loss']:.2f}")
    print(f"  حداکثر افت: ${m['max_dd']:,.0f} ({m['max_dd_pct']:.1%})")
    if res.halted:
        print(f"  ☠️ حساب در {res.halt_time} نابود شد (margin call) — "
              f"فقط {m['n_trades']:,} معامله از کل دوره رسید.")
