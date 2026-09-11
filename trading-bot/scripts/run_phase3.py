"""فاز ۳ — موتور ریسک در عمل: سایزینگ %-ریسک + نردبان بریکر.

اجرا:
    py scripts/run_phase3.py

سناریوها (سیگنال‌ها همه یکسان — فقط لایهٔ ریسک فرق می‌کند):
    E   مرجع: لات ثابت 0.01 (همان فاز ۲)
    G1  سایزینگ ۰.۵٪ ریسک، بدون بریکر
    G2  سایزینگ ۰.۵٪ + نردبان کامل بریکر — روی حساب $3,000
    G3  همان G2 روی حساب $30,000 (کفِ لات بروکر دیگر گلوگاه نیست)

سؤال علمی: نردبان ریسک، دنبالهٔ چپِ افت سرمایه (P(DD>15%) از MC)
را چقدر فشرده می‌کند — و به چه قیمتی؟
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.walkforward import bootstrap_maxdd
from bot.backtest.engine import BacktestConfig, Backtester, print_metrics
from bot.backtest.gate import GatedStrategy
from bot.backtest.risk import BreakerPolicy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, session_open
from bot.risk.circuit_breaker import CircuitBreaker


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۳: لایهٔ ریسک")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()

    print("═══ ۱) داده ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    offsets = infer_offset_minutes(mt5, duka)
    mt5_utc = shift_mt5_to_utc(mt5, offsets)
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)
    sess = session_open(m15["time"])
    print(f"  {len(m15):,} کندل M15 | رژیم/سشن مثل فاز ۲")

    inner = V0Strategy(h4, rr=2.5)   # همان پارامترهای E فاز ۲
    strat = GatedStrategy(inner, reg["regime"].to_numpy(), sess)

    print("\n═══ ۲) اجرا ═══")
    runs = {}
    # E — مرجع (لات ثابت)
    runs["E"] = Backtester(
        m15, GatedStrategy(V0Strategy(h4, rr=2.5), reg["regime"].to_numpy(), sess),
        BacktestConfig(fixed_lots=0.01, max_positions=1, cooldown_bars=1,
                       slippage_usd=0.05, spread_gate_usd=0.40)).run()
    print_metrics("E — مرجع: لات ثابت 0.01 (فاز ۲)", runs["E"])

    # G1 — فقط سایزینگ
    runs["G1"] = Backtester(
        m15, GatedStrategy(V0Strategy(h4, rr=2.5), reg["regime"].to_numpy(), sess),
        BacktestConfig(risk_pct=0.005, max_positions=1, cooldown_bars=1,
                       slippage_usd=0.05, spread_gate_usd=0.40)).run()
    print_metrics("G1 — سایزینگ ۰.۵٪ ریسک (بدون بریکر)", runs["G1"])

    policies = {}
    # نردبان تنظیم‌شده برای نرخ‌برد ~۳۳٪: derate از ۳ باخت (نه ۲) — رشتهٔ
    # ۲-باختی با این نرخ برد «عادی» است نه هشدار؛ ۶L و −۶٪ ماهانه ثابت
    tuned = dict(derate_at=3, deep_derate_at=4, pause_at=5, halt_at=6,
                 pause_hours=24.0, max_monthly_dd=0.06)
    for name, eq, brk in (("G2", 3000.0, CircuitBreaker()),
                          ("G3", 30000.0, CircuitBreaker()),
                          ("G4", 30000.0, CircuitBreaker(**tuned)),
                          ("G5", 3000.0, CircuitBreaker(**tuned))):
        pol = BreakerPolicy(brk)
        policies[name] = pol
        runs[name] = Backtester(
            m15, GatedStrategy(V0Strategy(h4, rr=2.5), reg["regime"].to_numpy(), sess),
            BacktestConfig(start_equity=eq, risk_pct=0.005, max_positions=1,
                           cooldown_bars=1, slippage_usd=0.05,
                           spread_gate_usd=0.40), risk_policy=pol).run()
        label = f"{name} — ۰.۵٪ + نردبان (حساب ${eq:,.0f})"
        if name in ("G4", "G5"):
            label += "، نردبان تنظیمی ۳/۴/۵/۶"
        print_metrics(label, runs[name])

    print("\n═══ ۳) رفتار نردبان ═══")
    for name, pol in policies.items():
        c = pol.counters
        m = runs[name].metrics
        print(f"  {name}: derate={c.get('derate', 0)} | deep={c.get('deep_derate', 0)} "
              f"| مکث={c.get('pause', 0)} | halt={c.get('halt', 0)} "
              f"| halt ماهانه={c.get('monthly_halt', 0)} | ack={c.get('ack', 0)} "
              f"| برگشت={c.get('restore', 0)}")
        print(f"      ورودهای بلاک‌شده={c.get('entry_blocks', 0)} | "
              f"کفِ لات={m.get('min_lot_clamps', 0)} بار | "
              f"ماه‌های halt: {pol.halted_months or '—'}")

    print("\n═══ ۴) مونت‌کارلو روی توالی معاملات (۲۰۰۰ مسیر) ═══")
    print(f"{'سناریو':<6}{'DD میانه':>10}{'DD p5':>9}{'P(DD<−۱۵٪)':>12}{'P(DD<−۲۵٪)':>12}{'eq پایانی میانه':>17}")
    for name in ("E", "G2", "G3", "G4", "G5"):
        res = runs[name]
        if res.trades.empty:
            continue
        mc = bootstrap_maxdd(res.trades["pnl"].to_numpy(), 2000, 42,
                             res.metrics.get("start_equity", 3000.0))
        print(f"{name:<6}{mc['maxdd_p50']:>10.1%}{mc['maxdd_p5']:>9.1%}"
              f"{mc['p_dd_below_15']:>12.1%}{mc['p_dd_below_25']:>12.1%}"
              f"{mc['terminal_p50']:>17,.0f}")

    print("\n═══ ۵) جمع‌بندی ═══")
    print(f"{'سناریو':<6}{'معاملات':>9}{'برد%':>8}{'PF':>7}{'سود$':>10}{'افت%':>9}")
    for name, res in runs.items():
        m = res.metrics
        if m.get("n_trades", 0) == 0:
            print(f"{name:<6}{0:>9}")
            continue
        print(f"{name:<6}{m['n_trades']:>9,}{m['win_rate']:>8.1%}"
              f"{m['profit_factor']:>7.2f}{m['total_pnl']:>10,.0f}"
              f"{m['max_dd_pct']:>9.1%}")

    save = find(pathlib.Path(args.mt5).name).parent
    try:
        for name, res in runs.items():
            if not res.trades.empty:
                res.trades.to_csv(save / f"phase3_{name}_trades.csv.gz",
                                  index=False, compression="gzip")
        print(f"\n💾 معاملات ذخیره شد در: {save}")
    except OSError as e:
        print(f"\n(ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
