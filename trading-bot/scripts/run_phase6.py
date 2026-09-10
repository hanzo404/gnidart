"""فاز ۶ — متا-لیبلینگ: فیلتر کیفیت در لحظه‌ی ورود.

اجرا:
    py scripts/run_phase6.py

فرایند:
    ۱. کاندیداها: همان استراتژی ولی بدون گیت رژیم (فقط سشن + فقط خرید،
       لات ثابت، بدون بریکر) — نتیجه: ~۱۹ کاندیدا در ماه با PF ~1.21 OOS
    ۲. دیتاست: فیچرهای علّیِ لحظه‌ی تصمیم + لیبل = نتیجه‌ی واقعی با هزینه
    ۳. Walk-Forward خالص‌شده: در هر پنجره، {take_all, logit, gbm}×آستانه
       فقط با ۱۲ ماه IS انتخاب می‌شود و روی ماه OOS منجمد اعمال می‌گردد
    ۴. مقایسه‌ی سه‌راهی OOS: گیت کامل (فعلی) | همه‌ی کاندیداها | متا-فیلتر
    ۵. حساسیت آستانه‌ی ثابت (بدون انتخاب پنجره‌ای) — چک پایداری

هدف آزمون (از جلسه‌ی طراحی): ۱۵–۲۵ معامله در ماه با PF ≥ 1.4 در OOS.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.meta import (FEATURES, MetaWFOConfig, agg, build_dataset,
                               fit_probs, run_meta_wfo)
from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, TREND_UP, session_open


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۶: متا-لیبلینگ")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    print("═══ ۱) داده ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka))
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    regime = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)
    fake_trend = np.full(len(m15), TREND_UP)
    months = (m15["time"].iloc[-1] - m15["time"].iloc[0]).days / 30.44

    def run_candidates(reg_arr):
        """لات ثابت + بدون بریکر → هر کاندیدا مستقل و قابل‌لیبل."""
        strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg_arr, sess,
                              direction=+1)
        return Backtester(m15, strat, BacktestConfig(
            start_equity=3000.0, fixed_lots=0.01, max_positions=1,
            cooldown_bars=1, slippage_usd=0.05,
            spread_gate_usd=0.40)).run()

    print("\n═══ ۲) کاندیداها (بدون گیت رژیم) و مرجعِ گیت کامل ═══")
    cand = run_candidates(fake_trend)
    ref = run_candidates(regime)
    print(f"  کاندیداها: {len(cand.trades)} | {len(cand.trades)/months:.1f}/ماه | "
          f"برد {(cand.trades['pnl'] > 0).mean():.1%} | "
          f"PF {cand.metrics['profit_factor']:.2f}")
    print(f"  گیت کامل:  {len(ref.trades)} | {len(ref.trades)/months:.1f}/ماه | "
          f"برد {(ref.trades['pnl'] > 0).mean():.1%} | "
          f"PF {ref.metrics['profit_factor']:.2f}")

    print("\n═══ ۳) دیتاست متا ═══")
    ds = build_dataset(m15, h4, regime, sess, cand.trades)
    print(f"  {len(ds)} سطر | {len(FEATURES)} فیچر | "
          f"برد {ds['win'].mean():.1%} | پراکندگی R: "
          f"{ds['r'].std():.2f}")

    print("\n═══ ۴) Walk-Forward خالص‌شده (IS=12ماه، انتخاب با IS-pnl) ═══")
    cfg = MetaWFOConfig()
    res = run_meta_wfo(ds, cfg)
    n_win = len(res.per_window)
    print(f"  پنجره‌های OOS: {n_win}")
    dist = res.per_window["chosen"].str.split("@").str[0].value_counts()
    for k, v in dist.items():
        print(f"    انتخاب: {k} — {v} ماه")

    # مرجع گیت کامل، فقط روی همان ماه‌های OOS
    oos_months = set()
    for lbl in res.per_window["oos_month"]:
        oos_months.update(lbl.split(","))
    ref_oos = ref.trades.assign(
        m=pd.to_datetime(ref.trades["entry_time"]).dt.strftime("%Y-%m"))
    ref_oos = ref_oos[ref_oos["m"].isin(oos_months)]

    print("\n═══ ۵) مقایسه‌ی سه‌راهی OOS (لات ثابت 0.01) ═══")
    for label, frame in (("گیت کامل (ربات فعلی)", ref_oos),
                         ("همه‌ی کاندیداها      ", res.oos_take_all),
                         ("متا-فیلتر انتخابی    ", res.oos_meta)):
        a = agg(frame, n_win)
        print(f"  {label}: {a['n']:4d}t | {a['per_month']:5.1f}/ماه | "
              f"برد {a['wr']:.1%} | PF {a['pf']:.2f} | ${a['pnl']:7,.0f}")

    print("\n═══ ۶) حساسیت آستانه‌ی ثابت (بدون انتخاب پنجره‌ای) ═══")
    from bot.analysis.meta import purged_monthly_windows
    for name, th in (("logit", 0.55), ("logit", 0.60),
                     ("gbm", 0.55), ("gbm", 0.60)):
        taken = []
        for is_idx, oos_idx, _ in purged_monthly_windows(ds, 12, 1):
            if len(is_idx) < cfg.min_kept_is:
                continue
            probs = fit_probs(ds, is_idx, oos_idx, cfg)
            if probs.get(name) is None:
                continue
            p_oos = probs[name][1]
            taken.append(ds.iloc[oos_idx][p_oos >= th])
        frame = pd.concat(taken, ignore_index=True) if taken else pd.DataFrame()
        a = agg(frame, n_win)
        print(f"  {name}@{th:.2f}: {a['n']:4d}t | {a['per_month']:5.1f}/ماه | "
              f"برد {a['wr']:.1%} | PF {a['pf']:.2f} | ${a['pnl']:7,.0f}")

    ds.to_csv(save / "phase6_candidates.csv", index=False)
    res.per_window.to_csv(save / "phase6_wfo_windows.csv", index=False)
    print(f"\n💾 marketdata/phase6_candidates.csv + phase6_wfo_windows.csv")
    print("\nقانون قضاوت: متا فقط وقتی می‌گذرد که PF_OOS(متا) ≥ 1.4 و "
          "فرکانس ≥ ۱۵/ماه — وگرنه گیت کامل می‌ماند.")


if __name__ == "__main__":
    main()
