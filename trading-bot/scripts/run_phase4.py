"""فاز ۴ — حلقهٔ خودتحلیلی: ربات بعد از هر رشتهٔ باخت «می‌بیند و ثبت می‌کند».

اجرا:
    py scripts/run_phase4.py

نمایش:
    ۱. همان بک‌تست G4 فاز ۳ (نردبان تنظیمی) را اجرا می‌کنیم
    ۲. توالی معاملات را «بازپخش» می‌کنیم: هر جا بریکر فعال می‌شد (رشتهٔ ۳)،
       همان لحظه گزارش تشخیصی می‌سازیم — فقط با دادهٔ تا آن لحظه
    ۳. پست‌مورتم نهایی روی کل نمونه (قضاوت آماری بزرگ)

نکتهٔ طراحی: این لایه هیچ چیزی در رفتار معامله تغییر نمی‌دهد —
عمداً. عدد نهایی G4 با و بدون این ماژول بایت‌به‌بایت یکی است.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.diagnostics import VERDICT_FA, diagnose_streak
from bot.analysis.postmortem import attach_regime, postmortem
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
    ap = argparse.ArgumentParser(description="فاز ۴: خودتحلیلی")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    ap.add_argument("--baseline-wr", type=float, default=0.348,
                    help="نرخ‌برد پایه از walk-forward OOS فاز ۲-ب")
    args = ap.parse_args()

    print("═══ ۱) داده + بک‌تست G4 (نردبان تنظیمی) ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    offsets = infer_offset_minutes(mt5, duka)
    mt5_utc = shift_mt5_to_utc(mt5, offsets)
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)
    sess = session_open(m15["time"])

    pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                       pause_at=5, halt_at=6))
    res = Backtester(
        m15, GatedStrategy(V0Strategy(h4, rr=2.5), reg["regime"].to_numpy(), sess),
        BacktestConfig(start_equity=30000.0, risk_pct=0.005, max_positions=1,
                       cooldown_bars=1, slippage_usd=0.05,
                       spread_gate_usd=0.40), risk_policy=pol).run()
    print_metrics("G4 — نردبان تنظیمی (بدون هیچ تغییری از فاز ۳)", res)

    # ---------- بازپخش تشخیصی ----------
    print("\n═══ ۲) بازپخش: ربات در هر رشتهٔ ۳-باختی چه می‌دید؟ ═══")
    trades = res.trades.reset_index(drop=True)
    reg_all = reg[["time", "regime"]]
    reports = []
    streak = 0
    for i in range(len(trades)):
        r = trades.loc[i]
        streak = streak + 1 if r["pnl"] <= 0 else 0
        if streak == 3:   # نقطهٔ فعال‌شدن derate
            so_far = trades.iloc[:i + 1]
            ts = pd.Timestamp(r["exit_time"])
            reg_until = reg_all[reg_all["time"] <= ts]   # علیّت: فقط گذشته
            rep = diagnose_streak(so_far, streak=3, regime_df=reg_until,
                                  baseline_wr=args.baseline_wr, ts=ts)
            reports.append(rep)

    print(f"  {len(reports)} نقطهٔ تشخیص در ۳ سال")
    if reports:
        dist = pd.Series([r.verdict for r in reports]).value_counts()
        for v, n in dist.items():
            print(f"    {VERDICT_FA[v]}: {n} بار")

        # دو نمونهٔ گزارش کامل: اولین و بدترین (طولانی‌ترین) رشته
        longest = max(reports, key=lambda r: r.stats.get("streak_span_hours") or 0)
        for label, rep in (("نمونه ۱ (اولین)", reports[0]),
                           ("نمونه ۲ (بدترین خوشه)", longest)):
            print(f"\n  ── {label}: {rep.ts} ──")
            for f in rep.findings:
                print(f"     • {f}")

    # ---------- پست‌مورتم نهایی ----------
    print("\n═══ ۳) پست‌مورتم آماری کل دوره (قضاوت روی نمونهٔ بزرگ) ═══")
    tr = attach_regime(trades, reg_all)
    pm = postmortem(tr, baseline_wr=args.baseline_wr)
    print(f"  توصیه: {pm['recommendation']} — {pm['reason']}")
    print(f"  PF={pm.get('profit_factor', float('nan')):.2f} | "
          f"انتظار={pm.get('expectancy_r', float('nan')):.3f}R | "
          f"نرخ‌برد={pm.get('win_rate', float('nan')):.1%}")
    if "by_direction" in pm:
        print("  شکستگی جهت:")
        for k, v in pm["by_direction"].items():
            print(f"    {k}: {v['count']} معامله، ${v['sum']:,.0f}")
    if "entry_regime" in tr.columns:
        g = tr.dropna(subset=["entry_regime"]).groupby("entry_regime")["pnl"] \
            .agg(["count", "sum"]).round(0)
        print("  شکستگی رژیمِ ورود:")
        from bot.regime.engine import NAMES
        for idx, row in g.iterrows():
            print(f"    {NAMES.get(idx, idx)}: {int(row['count'])} معامله، ${row['sum']:,.0f}")

    # ---------- آزمایش sandbox: حذف جهت زیان‌ده (پیشنهادِ لایهٔ تشخیص) ----------
    print("\n═══ ۴) آزمایش sandbox: حذف جهتِ زیان‌ده چه می‌شود؟ ═══")
    print("  (پیشنهاد از شکستگی پست‌مورتم — تغییرِ رفتار فقط از این مسیر مجاز است)")

    def _g(direction):
        pol2 = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                            pause_at=5, halt_at=6))
        strat2 = GatedStrategy(V0Strategy(h4, rr=2.5), reg["regime"].to_numpy(),
                               sess, direction=direction)
        return Backtester(
            m15, strat2,
            BacktestConfig(start_equity=30000.0, risk_pct=0.005,
                           max_positions=1, cooldown_bars=1, slippage_usd=0.05,
                           spread_gate_usd=0.40), risk_policy=pol2).run()

    for label, d in (("G6 — فقط خرید", +1), ("G7 — فقط فروش (شاهد)", -1)):
        m6 = _g(d).metrics
        print(f"  {label}: {m6['n_trades']} معامله | برد {m6['win_rate']:.1%} | "
              f"PF {m6['profit_factor']:.2f} | ${m6['total_pnl']:,.0f} | "
              f"DD {m6['max_dd_pct']:.1%} | {m6['expectancy_r']:.3f}R")

    r6 = _g(+1)
    t6 = r6.trades.assign(
        year=pd.to_datetime(r6.trades["entry_time"]).dt.year)
    print("  سال‌به‌سال G6 (هشدار تمرکز سود):")
    print(t6.groupby("year")["pnl"].agg(["count", "sum"]).round(0).to_string())
    print("  ⚠️ صادقانه: سود G6 عمدتاً از سال ۲۰۲۵ (بازار صعودی تاریخی) است؛")
    print("     ۲۰۲۶ (سقف+ریزش) تا الان منفی. این قابلیت را در دمو ماه‌به‌ماه")
    print("     زیر نظر پست‌مورتم نگه می‌داریم؛ بریکر −۶٪ هم شبکهٔ برق است.")

    save = find(pathlib.Path(args.mt5).name).parent
    try:
        pd.DataFrame([{"ts": r.ts, "streak": r.streak, "verdict": r.verdict,
                       "confidence": r.confidence,
                       "findings": " | ".join(r.findings)} for r in reports]) \
            .to_csv(save / "phase4_diagnostic_reports.csv", index=False)
        print(f"\n💾 گزارش‌ها ذخیره شد در: {save}")
    except OSError as e:
        print(f"\n(ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
