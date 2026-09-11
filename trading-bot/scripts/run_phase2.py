"""فاز ۲ — موتور رژیم + فیلتر سشن؛ همان استراتژی، فقط «وقتی درست است».

اجرا:
    py scripts/run_phase2.py

سناریوها (همه با هزینه‌های کامل، مثل فاز ۱):
    C  پایه: v0 روی M15 (مرجع مقایسه)
    E  همان C + گیت رژیم (روند هم‌جهت) + گیت سشن (۱۲–۲۰ UTC)
    D/F همان دو، روی منبع مستقل Dukascopy

سؤال علمی این فاز: آیا «نگه‌داشتن PF و کاهش افت سرمایه به زیر ۱۰٪»
با فیلتر کردن معاملات بی‌رژیم ممکن است؟
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester, print_metrics
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (attach_spread_profile, infer_offset_minutes,
                              load_duka, load_mt5, resample_tf,
                              shift_mt5_to_utc, spread_profile_by_hour, to_m15)
from bot.regime.engine import NAMES, RegimeEngine, session_open


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def regime_stats(reg: pd.DataFrame) -> None:
    dist = reg["regime"].map(NAMES).value_counts(normalize=True)
    print("  توزیع رژیم (سهم کندل‌های M15):")
    for k in ["TREND_UP", "TREND_DOWN", "RANGE", "CHAOS"]:
        if k in dist.index:
            print(f"    {k:<11} {dist[k]:.1%}")


def pnl_by_regime(trades: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """سود/زیان معاملاتِ پایه، گروه‌بندی بر اساس رژیمِ لحظهٔ ورود —
    پاسخ داده به «آیا معاملات بی‌رژیم ضررده بودند؟»"""
    r = reg[["time", "regime"]]
    t = trades.merge(r, left_on="entry_time", right_on="time", how="left")
    g = t.groupby(t["regime"].map(NAMES)).agg(
        تعداد=("pnl", "size"), سود=("pnl", "sum"),
        میانگین=("pnl", "mean")).round(1)
    return g.sort_values("سود", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۲: رژیم + سشن")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()

    print("═══ ۱) داده ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    offsets = infer_offset_minutes(mt5, duka)
    mt5_utc = shift_mt5_to_utc(mt5, offsets)
    prof = spread_profile_by_hour(mt5_utc)
    m15_mt5 = to_m15(mt5_utc)                      # UTC — برای فیلتر سشن
    m15_duka = attach_spread_profile(to_m15(duka), prof)
    h4_mt5 = resample_tf(mt5_utc, "4h")
    h4_duka = resample_tf(duka, "4h")
    print(f"  MT5(M15,UTC)={len(m15_mt5):,} | duka(M15,UTC)={len(m15_duka):,}")

    print("\n═══ ۲) رژیم بازار (MT5-UTC) ═══")
    eng = RegimeEngine()
    reg_mt5 = eng.compute(m15_mt5)
    regime_stats(reg_mt5)
    reg_duka = eng.compute(m15_duka)

    sess_mt5 = session_open(m15_mt5["time"])
    sess_duka = session_open(m15_duka["time"])
    print(f"\n  سشن ورود ۱۲–۲۰ UTC: {sess_mt5.mean():.0%} کندل‌ها باز است")

    print("\n═══ ۳) شبیه‌سازی‌ها ═══")
    cfg = BacktestConfig(fixed_lots=0.01, max_positions=1, cooldown_bars=1,
                         slippage_usd=0.05, spread_gate_usd=0.40)

    def run_pair(bars, h4, regime_df, sess, gated: bool):
        inner = V0Strategy(h4)
        strat = (GatedStrategy(inner, regime_df["regime"].to_numpy(), sess)
                 if gated else inner)
        return Backtester(bars, strat, cfg).run()

    resC = run_pair(m15_mt5, h4_mt5, reg_mt5, sess_mt5, gated=False)
    resE = run_pair(m15_mt5, h4_mt5, reg_mt5, sess_mt5, gated=True)
    resD = run_pair(m15_duka, h4_duka, reg_duka, sess_duka, gated=False)
    resF = run_pair(m15_duka, h4_duka, reg_duka, sess_duka, gated=True)

    for title, res in [("C — پایه (M15، بدون رژیم)", resC),
                       ("E — پایه + رژیم + سشن (MT5)", resE),
                       ("D — پایه روی duka", resD),
                       ("F — رژیم + سشن روی duka", resF)]:
        print_metrics(title, res)

    print("\n═══ ۴) پاسخ به سؤال علمی: سود C بر حسب رژیمِ لحظهٔ ورود ═══")
    if not resC.trades.empty:
        print(pnl_by_regime(resC.trades, reg_mt5).to_string())

    print("\n═══ ۵) جمع‌بندی ═══")
    print(f"{'سناریو':<8}{'معاملات':>9}{'برد%':>8}{'PF':>7}{'سود$':>10}{'افت%':>9}")
    for k, res in [("C", resC), ("E", resE), ("D", resD), ("F", resF)]:
        m = res.metrics
        if m.get("n_trades", 0) == 0:
            print(f"{k:<8}{0:>9}")
            continue
        print(f"{k:<8}{m['n_trades']:>9,}{m['win_rate']:>8.1%}"
              f"{m['profit_factor']:>7.2f}{m['total_pnl']:>10,.0f}"
              f"{m['max_dd_pct']:>9.1%}")

    # تحلیل ماهانه/جهت برای سناریوی گیت‌شده
    if not resE.trades.empty:
        t = resE.trades
        t = t.assign(month=t["entry_time"].dt.to_period("M"),
                     dirn=t["direction"].map({1: "خرید", -1: "فروش"}))
        m = t.groupby("month")["pnl"].sum()
        print(f"\n  E: ماه‌های سودده {(m > 0).sum()} از {len(m)} | "
              f"بدترین ماه ${m.min():,.0f}")
        print("  E بر حسب جهت:")
        print(t.groupby("dirn")["pnl"].agg(["count", "sum"]).round(0).to_string())

    save = find(pathlib.Path(args.mt5).name).parent
    try:
        for k, res in [("C", resC), ("E", resE), ("D", resD), ("F", resF)]:
            if not res.trades.empty:
                res.trades.to_csv(save / f"phase2_{k}_trades.csv.gz",
                                  index=False, compression="gzip")
        print(f"\n💾 معاملات ذخیره شد در: {save}")
    except OSError as e:
        print(f"\n(ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
