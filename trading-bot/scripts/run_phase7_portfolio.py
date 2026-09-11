"""فاز ۷ — پورتفوی طلا + نقره: «موقعیت‌های بیشتر» بالاخره اندازه‌گیری می‌شود.

اجرا:
    py scripts/run_phase7_portfolio.py

سؤال: دو آستینِ مستقل (طلا G6 + نقره‌ی گیت‌شده) روی یک پنجرهٔ سه‌ساله،
هر دو با لات ثابت 0.01 روی $30k:
    - فرکانس ترکیبی چند می‌شود؟ (هدف پروژه از روز اول: معاملهٔ بیشتر)
    - هم‌زمانی ورودها چقدر است؟ (دو پوزیشن هم‌زمان = ریسک هم‌زمان)
    - ماه‌های هر دو منفی، هم‌پوشانی افت‌ها، PF و DD پورتفوی
    - خط‌کش بریکر: بدترین رشتهٔ ماه‌های منفیِ ترکیبی

هزینه‌ها — طلا: اسپرد واقعی MT5 (ستون spread_usd)؛ نقره: فرض ثابت
$0.018 (اندازه‌گیری‌شده از دموی MetaQuotes: ۱۸ پوینت، ۳ رقم اعشار) —
با تکرار روی $0.04 برای محک‌کاری.
بدون بریکر (سنجش لبه است)؛ هر دو فقط-خریدِ گیت‌شده؛ تنظیم‌شده روی طلا،
دست‌نخورده روی نقره (کل نمونهٔ نقره OOS است).
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
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, session_open


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def run_gold(mt5_path, duka_path):
    mt5 = load_mt5(mt5_path)
    duka = load_duka(duka_path)
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka))
    m15, h4 = to_m15(mt5_utc), resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)
    strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg, sess, direction=+1)
    res = Backtester(m15, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, max_positions=1,
        cooldown_bars=1, slippage_usd=0.05, spread_gate_usd=0.40)).run()
    return res.trades.assign(symbol="XAUUSD")


def run_silver(sil_path, spread):
    sil = load_duka(sil_path)
    s15, s4 = to_m15(sil), resample_tf(sil, "4h")
    reg = RegimeEngine().compute(s15)["regime"].to_numpy()
    sess = np.asarray(session_open(s15["time"]), dtype=bool)
    m = s15.copy()
    m["spread_usd"] = spread
    strat = GatedStrategy(V0Strategy(s4, rr=2.5, sl_pad=0.01093),
                          reg, sess, direction=+1)
    res = Backtester(m, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, contract_oz=5000.0,
        max_positions=1, cooldown_bars=1, slippage_usd=0.00109,
        spread_gate_usd=None)).run()
    return res.trades.assign(symbol="XAGUSD")


def stats(t: pd.DataFrame, label: str, months: float):
    if not len(t):
        print(f"  {label}: 0 معامله")
        return
    gw = t.loc[t["pnl"] > 0, "pnl"].sum()
    gl = -t.loc[t["pnl"] <= 0, "pnl"].sum()
    eq = t.sort_values("entry_time")["pnl"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak).min()
    print(f"  {label:22s} {len(t):4d}t | {len(t)/months:5.1f}/ماه | "
          f"برد {(t['pnl'] > 0).mean():.1%} | PF {gw/gl:.2f} | "
          f"${t['pnl'].sum():8,.0f} | بدترین افت ${-dd:,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۷: پورتفوی طلا+نقره")
    ap.add_argument("--gold-mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--gold-duka", default="data/xauusd_m1_duka.csv.gz")
    ap.add_argument("--silver", default="data/xagusd_m1_duka.csv.gz")
    ap.add_argument("--silver-spread", type=float, default=0.018)
    args = ap.parse_args()
    save = find(pathlib.Path(args.gold_mt5).name).parent

    g = run_gold(find(pathlib.Path(args.gold_mt5).name),
                 find(pathlib.Path(args.gold_duka).name))
    s = run_silver(find(pathlib.Path(args.silver).name), args.silver_spread)
    months = 36.0

    print(f"═══ پورتفوی طلا + نقره (لات ثابت 0.01، $30k، ۳ سال، بدون بریکر) ═══")
    print(f"  (اسپرد نقره ${args.silver_spread} — اندازه‌گیری دموی تو)")
    stats(g, "طلا (G6)", months)
    stats(s, "نقره", months)
    both = pd.concat([g, s], ignore_index=True).sort_values("entry_time")
    stats(both, "پورتفوی ترکیبی", months)

    # هم‌زمانی ورودها
    ge = set(g["entry_time"])
    se = set(s["entry_time"])
    overlap = ge & se
    pct = len(overlap) / len(ge | se) if (ge | se) else 0.0
    print(f"\n  ورودهای هم‌کندل (هر دو نماد هم‌لحظه): {len(overlap)} "
          f"({pct:.0%} از لحظات ورود)")

    # ماه‌به‌سال
    mo = both.assign(m=pd.to_datetime(both["entry_time"]).dt.to_period("M")) \
        .groupby(["m", "symbol"])["pnl"].sum().unstack(fill_value=0.0)
    mo["合计"] = mo.sum(axis=1)
    pos = (mo["合计"] > 0).mean()
    both_neg = ((mo.get("XAUUSD", 0) <= 0) & (mo.get("XAGUSD", 0) <= 0)).mean()
    neg = (mo["合计"] <= 0).astype(int)
    streak = neg * (neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1)
    print(f"  ماه‌های مثبتِ ترکیبی: {pos:.0%} | ماه‌هایی که هر دو منفی بودند: "
          f"{both_neg:.0%} | بدترین رشتهٔ منفی: {streak.max()} ماه")
    corr = mo["XAUUSD"].corr(mo["XAGUSD"]) if len(mo) > 2 else float("nan")
    print(f"  هم‌بستگی سود ماهانه طلا↔نقره: {corr:+.2f}")
    print("\n  سال‌به‌سال (دلار):")
    yr = both.assign(y=pd.to_datetime(both["entry_time"]).dt.year) \
        .pivot_table(index="y", columns="symbol", values="pnl",
                     aggfunc="sum").round(0)
    print(yr.to_string())

    try:
        mo.assign(total=mo["合计"]).to_csv(save / "phase7_portfolio_monthly.csv")
        pd.DataFrame([{"gold_trades": len(g), "silver_trades": len(s),
                       "combined_pm": round(len(both) / months, 1),
                       "combined_pf": round(
                           both.loc[both['pnl'] > 0, 'pnl'].sum()
                           / -both.loc[both['pnl'] <= 0, 'pnl'].sum(), 2),
                       "monthly_corr": round(corr, 2),
                       "pct_months_pos": round(pos, 3)}]).to_csv(
            save / "phase7_portfolio_summary.csv", index=False)
        print(f"\n💾 {save}/phase7_portfolio_monthly.csv + phase7_portfolio_summary.csv")
    except OSError as e:
        print(f"  (ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
