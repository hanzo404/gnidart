"""فاز ۲-ب — Walk-Forward: آیا نتایج فاز ۲ بعد از لایهٔ صداقت سرِ جایشان می‌مانند؟

اجرا:
    py scripts/run_phase2b.py

فرایند: برای هر ماهِ آزمون، پارامترها فقط با ۱۲ ماه قبلش انتخاب می‌شوند
(گرید ۲۷ ترکیبی از ADX/CHAOS/RR) و همان ماه با پارامتر منجمد معامله می‌شود.
مقایسهٔ مرجع: همان ماه‌ها با پارامترهای ثابتِ پیش‌فرض (بدون هیچ بهینه‌سازی).
جمع‌بندی نهایی: مونت‌کارلو روی توالی معاملات OOS → توزیع افت سرمایه.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.walkforward import WFOConfig, WalkForwardOptimizer
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def show(label: str, m: dict) -> None:
    if m.get("n_trades", 0) == 0:
        print(f"  {label}: هیچ معامله‌ای")
        return
    print(f"  {label}: {m['n_trades']:,} معامله | برد {m['win_rate']:.1%} | "
          f"PF {m['profit_factor']:.2f} | سود ${m['total_pnl']:,.0f} | "
          f"افت(ترید-سطح) {m['trade_level_maxdd_pct']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۲-ب: walk-forward")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    ap.add_argument("--is-months", type=int, default=12)
    ap.add_argument("--oos-months", type=int, default=1)
    args = ap.parse_args()

    print("═══ ۱) داده (MT5 → UTC) ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    offsets = infer_offset_minutes(mt5, duka)
    mt5_utc = shift_mt5_to_utc(mt5, offsets)
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    print(f"  {len(m15):,} کندل M15 | {m15['time'].min()} → {m15['time'].max()}")

    print(f"\n═══ ۲) Walk-Forward (IS={args.is_months}ماه، OOS={args.oos_months}ماه، "
          f"گرید ۲۷ ترکیب) ═══")
    cfg = WFOConfig(is_months=args.is_months, oos_months=args.oos_months)
    wfo = WalkForwardOptimizer(m15, h4, cfg)
    res = wfo.run()

    pw = res.per_window
    print(f"  پنجره‌های OOS: {len(pw)} | از {pw['oos_month'].iloc[0]} تا "
          f"{pw['oos_month'].iloc[-1]}")

    print("\n═══ ۳) نتیجهٔ صادقانه (OOS تجمیعی) ═══")
    show("Walk-Forward (انتخاب ماهانه)", res.metrics)
    show("پارامتر ثابت پیش‌فرض   ", res.static_metrics)

    print("\n═══ ۴) پایداری انتخاب پارامتر ═══")
    if len(res.chosen_freq):
        top = res.chosen_freq.head(5)
        for _, row in top.iterrows():
            print(f"    {row['params']:<28} {row['count']} بار "
                  f"({row['count']/len(pw):.0%})")

    print("\n═══ ۵) ماه‌به‌ماه (WFO در برابر ثابت) ═══")
    pw2 = pw.copy()
    pw2["oos_month"] = pw2["oos_month"].str.slice(0, 7)
    with pd.option_context("display.max_rows", 100):
        print(pw2[["oos_month", "trades", "pnl", "static_pnl"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    print("\n═══ ۶) مونت‌کارلو روی توالی معاملات OOS (۲۰۰۰ مسیر) ═══")
    mc = res.mc
    if mc:
        print(f"  افت سرمایه: p5 {mc['maxdd_p5']:.1%} | میانه {mc['maxdd_p50']:.1%} "
              f"| p95 {mc['maxdd_p95']:.1%}")
        print(f"  equity پایانی: p5 ${mc['terminal_p5']:,.0f} | میانه "
              f"${mc['terminal_p50']:,.0f} | p95 ${mc['terminal_p95']:,.0f}")
        print(f"  احتمال افت بدتر از ۱۵٪: {mc['p_dd_below_15']:.1%} | "
              f"بدتر از ۲۵٪: {mc['p_dd_below_25']:.1%}")

    save = find(pathlib.Path(args.mt5).name).parent
    try:
        res.oos_trades.to_csv(save / "phase2b_wfo_oos_trades.csv.gz",
                              index=False, compression="gzip")
        res.per_window.to_csv(save / "phase2b_wfo_windows.csv",
                              index=False)
        print(f"\n💾 خروجی ذخیره شد در: {save}")
    except OSError as e:
        print(f"\n(ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
