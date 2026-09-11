"""تست تایم‌فریم — «ربات ما را اسکلپر کنیم؟»

اجرا:
    py scripts/run_phase5c_timeframe.py

سؤال: همین استراتژی (FVG + بایاس H4 + رژیم + سشن + فقط-خرید + بریکر)
روی تایم‌فریم‌های سریع‌تر چه می‌شود؟ M15 (فعلی) در برابر M5 و M1.

چرا این تست صادقانه است:
    - استراتژی/گیت‌ها/ریسک بایت‌به‌بایت یکی‌اند؛ فقط تایم‌فریم عوض می‌شود
    - اسپردِ واقعیِ ثبت‌شدهٔ بروکر در هر کندل اعمال می‌شود (نه فرض صفر)
    - سه سناریوی هزینه: ثبت‌شده | +$0.10 | +$0.20 (مارک‌آپ بدتر بروکر)

فرضیهٔ پیش از تست (که خود آزمایش نقضش می‌کند یا تأیید):
    کندل‌های سریع‌تر → استاپ‌های تنگ‌تر → سهم اسپرد از سودِ هر معامله
    بالاتر → انتظارِ خالص سریع‌تر نابود می‌شود.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.risk import BreakerPolicy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc)
from bot.regime.engine import RegimeEngine, session_open
from bot.risk.circuit_breaker import CircuitBreaker


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def to_tf(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """ریسمپل عمومی با نگه‌داشتن اسپرد اولین دقیقه (مثل to_m15)."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum", "spread_usd": "first"}
    return (m1.set_index("time")
            .resample(rule, label="left", closed="left")
            .agg(agg).dropna(subset=["open"]).reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description="تست تایم‌فریم M15/M5/M1")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    m1 = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka))
    h4 = resample_tf(m1, "4h")
    months = (m1["time"].iloc[-1] - m1["time"].iloc[0]).days / 30.44
    print(f"داده: {months:.1f} ماه M1 | اسپرد میانهٔ ثبت‌شده: "
          f"${m1['spread_usd'].median():.2f}")

    rows = []
    for tf in ("15min", "5min", "1min"):
        bars = to_tf(m1, tf)
        regime = RegimeEngine().compute(bars)["regime"].to_numpy()
        sess = np.asarray(session_open(bars["time"]), dtype=bool)
        base_spread = bars["spread_usd"].to_numpy(float)

        for stress in (0.0, 0.10, 0.20):
            b = bars.copy()
            b["spread_usd"] = base_spread + stress
            pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                               pause_at=5, halt_at=6))
            strat = GatedStrategy(V0Strategy(h4, rr=2.5), regime, sess,
                                  direction=+1)
            res = Backtester(b, strat, BacktestConfig(
                start_equity=30000.0, risk_pct=0.005, max_positions=1,
                cooldown_bars=1, slippage_usd=0.05,
                spread_gate_usd=0.40), risk_policy=pol).run()
            m = res.metrics
            label = f"{tf:>6} | اسپرد {'ثبت‌شده' if stress == 0 else f'+${stress:.2f}'}"
            print(f"  {label:24s} {m['n_trades']:5d}t | {m['n_trades']/months:6.1f}/ماه | "
                  f"برد {m['win_rate']:.1%} | PF {m['profit_factor']:.2f} | "
                  f"${m['total_pnl']:8,.0f} | DD {m['max_dd_pct']:5.1%} | "
                  f"{m['expectancy_r']:+.3f}R")
            rows.append({"tf": tf, "spread_stress": stress,
                         "n": m["n_trades"],
                         "per_month": round(m["n_trades"] / months, 1),
                         "wr": round(m["win_rate"], 3),
                         "pf": round(m["profit_factor"], 2),
                         "pnl": round(m["total_pnl"]),
                         "dd": round(m["max_dd_pct"], 3),
                         "exp_r": round(m["expectancy_r"], 3)})

    pd.DataFrame(rows).to_csv(save / "timeframe_cost_matrix.csv", index=False)
    print(f"\n💾 marketdata/timeframe_cost_matrix.csv")


if __name__ == "__main__":
    main()
