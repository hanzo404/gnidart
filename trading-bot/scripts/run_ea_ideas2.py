"""آزمایش تکهٔ دومِ EA خارجی — سپتامبر ۲۰۲۶.

فرAGMENT این‌بار: تاریخچهٔ معاملات + سایزینگ لات + گارد ریسک (DD شناور) +
شمارندهٔ باخت روزانه + فیلتر سشن لندن/NY/آسیا.

دو ایدهٔ قابل‌آزمایش با دادهٔ خودمان:

    ۱) نقشهٔ سشن EA (آسیا ۰–۸ و ۲۲–۲۴، لندن ۷–۱۶، نیویورک ۱۳–۲۱ GMT)
       در برابر پنجرهٔ ۱۲–۲۰ UTC خودمان: شکست معاملات G6 بر اساس ساعت
       ورود UTC — آیا تقسیم‌بندی ریزترِ EA لبه‌ای اضافه می‌دهد؟
       (تشخیصِ درون-نمونه؛ تغییر گیت زنده فقط با walk-forward.)
    ۲) ریسکِ کفِ مین-لات روی حساب کوچک: وقتی لاتِ محاسبه‌شده از ۰.۰۱ کمتر
       شود، هم EA و هم رانرِ ما به ۰.۰۱ «گِرد می‌کنند» — یعنی ریسکِ واقعی
       از هدف (۰.۵٪) بیشتر می‌شود. روی $30k این هرگز رخ نمی‌دهد؛ روی دموی
       $3k بسته به فاصلهٔ استاپ رخ می‌دهد. اینجا دقیق می‌شماریم.

خروجی: marketdata/ea_ideas2_results.csv
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
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, session_open
from bot.risk.circuit_breaker import CircuitBreaker


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def session_bucket(hour_utc: int) -> str:
    """نقشهٔ سشن EA (GMT=UTC): آسیا ۰–۸ و ۲۲–۲۴ | لندن ۷–۱۶ | نیویورک ۱۳–۲۱."""
    if hour_utc >= 22 or hour_utc < 7:
        return "آسیا (22–07)"
    if 7 <= hour_utc < 12:
        return "لندن تنها (07–12)"
    if 12 <= hour_utc < 13:
        return "لندن دیرهنگام (12–13)"
    if 13 <= hour_utc < 16:
        return "هم‌پوشان لندن+NY (13–16)"
    if 16 <= hour_utc < 21:
        return "نیویورک (16–21)"
    return "خارج (21–22)"


def main() -> None:
    ap = argparse.ArgumentParser(description="آزمایش تکهٔ دوم EA")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka_m1 = load_duka(find(pathlib.Path(args.duka).name))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka_m1))
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)
    months = (m15["time"].iloc[-1] - m15["time"].iloc[0]).days / 30.44

    print("═══ G6 پایه (همان پیکربندی رسمی) ═══")
    cfg = BacktestConfig(start_equity=30000.0, risk_pct=0.005, max_positions=1,
                         cooldown_bars=1, slippage_usd=0.05,
                         spread_gate_usd=0.40)
    pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                       pause_at=5, halt_at=6))
    strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg, sess, direction=+1)
    res = Backtester(m15, strat, cfg, risk_policy=pol).run()
    m = res.metrics
    print(f"  {m['n_trades']}t | {m['n_trades']/months:.1f}/ماه | برد {m['win_rate']:.1%} | "
          f"PF {m['profit_factor']:.2f} | ${m['total_pnl']:,.0f} | DD {m['max_dd_pct']:.1%}")

    t = res.trades.copy()
    t["hour"] = pd.to_datetime(t["entry_time"]).dt.hour
    t["dist"] = (t["entry"] - t["stop"]).abs()
    t["sess"] = t["hour"].map(session_bucket)

    # ---------- ۱) شکست سشن (نقشهٔ EA در برابر پنجرهٔ ما) ----------
    print("\n═══ ۱) معاملات G6 بر اساس ساعت ورود UTC (نقشهٔ سشن EA) ═══")
    print("  (گیت ما فقط ۱۲–۲۰ را گذرانده — پس آسیا/لندنِ صبح ذاتاً غایب است)")
    rows = []
    g = t.groupby("sess")["pnl"].agg(["count", "sum"])
    for name, row in g.iterrows():
        sub = t[t["sess"] == name]
        print(f"  {name:26s} {int(row['count']):4d}t | برد {(sub['pnl'] > 0).mean():.1%} | "
              f"میانگین {sub['r'].mean():+.3f}R | ${row['sum']:,.0f}")
        rows.append({"bucket": name, "n": int(row["count"]),
                     "wr": round((sub["pnl"] > 0).mean(), 3),
                     "avg_r": round(sub["r"].mean(), 3),
                     "pnl": round(row["sum"])})

    # ساعت‌به‌ساعت داخل پنجرهٔ خودمان
    print("  ساعت‌به‌ساعت داخل پنجرهٔ ۱۲–۲۰:")
    for h in range(12, 20):
        sub = t[t["hour"] == h]
        if len(sub):
            print(f"    {h:02d}:00  {len(sub):3d}t | برد {(sub['pnl'] > 0).mean():.1%} | "
                  f"میانگین {sub['r'].mean():+.3f}R | ${sub['pnl'].sum():,.0f}")

    # ---------- ۲) ریسک کف مین-لات روی دموی $3k ----------
    print("\n═══ ۲) کف مین-لات: ریسک واقعی روی حساب $3,000 (هدف ۰.۵٪ = $15) ═══")
    print("  قانون EA/رانر ما: لات < 0.01 → 0.01 (یعنی ریسکِ بیشتر از هدف)")
    eq3k, risk_pct3k, min_lot, step, cap = 3000.0, 0.005, 0.01, 0.01, 0.10
    target = eq3k * risk_pct3k
    ideal = target / (t["dist"] * 100.0)
    actual = np.floor(ideal / step) * step
    actual = np.maximum(actual, min_lot)
    actual = np.minimum(actual, cap)
    real_risk = t["dist"] * actual * 100.0
    ratio = real_risk / target
    t["risk3k"] = real_risk

    print(f"  فاصلهٔ استاپ: میانه ${t['dist'].median():.1f} | میانگین ${t['dist'].mean():.1f} | "
          f"p90 ${t['dist'].quantile(0.9):.1f} | بیشینه ${t['dist'].max():.1f}")
    print(f"  معاملات با ریسکِ بیش از هدف (dist>$15): {(t['dist'] > 15).mean():.0%}")
    print(f"  معاملات با ریسک > 1.5× هدف: {(ratio > 1.5).mean():.0%} | "
          f"> 2× هدف: {(ratio > 2.0).mean():.0%}")
    print(f"  بدترین ریسکِ یک معامله: ${real_risk.max():.0f} = "
          f"{real_risk.max()/eq3k:.2%}ِ حساب")
    skip15 = (ratio > 1.5).sum()
    print(f"  اگر قانون «رد اگر ریسک مین-لات > 1.5× هدف» می‌بود: "
          f"{skip15} از {len(t)} معامله رد می‌شد ({skip15/len(t):.0%})")

    # سود/زیان کوهورت «رد‌شده تحت قانون 1.5×» — برای تصمیم آگاهانه
    sel = ratio > 1.5
    if sel.any():
        sub = t[sel]
        print(f"  کوهورتِ رد‌شدنی (ریسک>1.5×): {len(sub)}t | برد {(sub['pnl'] > 0).mean():.1%} | "
              f"میانگین {sub['r'].mean():+.3f}R | جمع ${sub['pnl'].sum():,.0f}")
    by_year = t.groupby(pd.to_datetime(t["entry_time"]).dt.year)["dist"]
    print("  روند فاصلهٔ استاپ سال‌به‌سال (طلا پرنوسان‌تر شده):")
    for y, s in by_year:
        print(f"    {y}: میانه ${s.median():.1f} | p90 ${s.quantile(0.9):.1f} | "
              f"ریسک واقعیِ $3k میانه ${t.loc[s.index, 'risk3k'].median():.0f}")

    rows.append({"bucket": "min_lot_overshoot_gt1.5x",
                 "n": int(skip15), "wr": None, "avg_r": None,
                 "pnl": int((ratio > 1.5).sum() / len(t) * 100)})
    pd.DataFrame(rows).to_csv(save / "ea_ideas2_results.csv", index=False)
    print(f"\n💾 ذخیره شد: {save / 'ea_ideas2_results.csv'}")


if __name__ == "__main__":
    main()
