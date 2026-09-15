"""مرز فرکانس–کیفیت — پاسخ آزمایشی به «معاملهٔ بیشتر بدون افت کیفیت؟»

اجرا:
    py scripts/run_phase5b_frontier.py

چهار سؤال، چهار بخش:
    ۱. اگر گیت‌ها را یکی‌یکی برداریم، فرکانس و کیفیت چه می‌شود؟ (A/B/C/D)
    ۲. یافتهٔ غیرمنتظرهٔ C (بدون گیت رژیم) سال‌به‌سال و روی منبع
       مستقل Dukascopy هم برقرار است؟
    ۳. صداقت نهایی: walk-forward دو-بازویی — برای هر ماهِ آزمون، انتخابِ
       «گیت روشن/خاموش» فقط با ۱۲ ماه قبلش انجام می‌شود.
    ۴. خروجی‌ها → marketdata/frontier_frequency_quality.csv و
       marketdata/wfo_gate_choice_per_window.csv

نتیجهٔ کلیدی (۲۰۲۶-۰۹): حذف گیت رژیم برای خریدها فرکانس را ~۲.۴ برابر
می‌کند و سود خالص OOS کمی بالاتر می‌رود، اما PF و کیفیتِ هر معامله
قابل‌توجه پایین می‌آید. یعنی «هم تعداد هم کیفیت» از این مسیر به‌دست
نمی‌آید — مسیر درستش متا-لیبلینگ (فاز ۶) است: کاندیداهای بیشتر +
فیلتر کیفیت در لحظهٔ ورود.
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
from bot.regime.engine import RegimeEngine, TREND_UP, session_open
from bot.risk.circuit_breaker import CircuitBreaker
from bot.backtest.risk import BreakerPolicy
from bot.analysis.walkforward import monthly_windows


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def main() -> None:
    ap = argparse.ArgumentParser(description="مرز فرکانس–کیفیت")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka_m1 = load_duka(find(pathlib.Path(args.duka).name))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka_m1))
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    regime = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)
    n = len(m15)
    months = (m15["time"].iloc[-1] - m15["time"].iloc[0]).days / 30.44
    print(f"داده: {months:.1f} ماه | {n:,} کندل M15")

    fake_trend = np.full(n, TREND_UP)     # گیت رژیم را کاملاً خنثی می‌کند
    all_open = np.ones(n, dtype=bool)     # گیت سشن را کاملاً خنثی می‌کند

    def run(reg_arr, sess_arr, eq=30000.0, risk=True):
        bt = BacktestConfig(start_equity=eq,
                            risk_pct=0.005 if risk else None,
                            fixed_lots=None if risk else 0.01,
                            max_positions=1, cooldown_bars=1,
                            slippage_usd=0.05, spread_gate_usd=0.40)
        pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                           pause_at=5, halt_at=6)) if risk else None
        strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg_arr, sess_arr,
                              direction=+1)
        return Backtester(m15, strat, bt, risk_policy=pol).run()

    # ---------- ۱) جدول مرز ---------- #
    print("\n═══ ۱) مرز فرکانس–کیفیت (۳ سال، فقط-خرید، نردبان بریکر) ═══")
    rows = []
    for label, r, s in (
            ("A: پایه (رژیم + سشن) = فعلی", regime, sess),
            ("B: بدون گیت سشن (24 ساعته)", regime, all_open),
            ("C: بدون گیت رژیم", fake_trend, sess),
            ("D: بدون هیچ گیت", fake_trend, all_open)):
        m = run(r, s).metrics
        print(f"  {label:32s} {m['n_trades']:4d}t | {m['n_trades']/months:5.1f}/ماه | "
              f"برد {m['win_rate']:.1%} | PF {m['profit_factor']:.2f} | "
              f"${m['total_pnl']:8,.0f} | DD {m['max_dd_pct']:5.1%} | "
              f"{m['expectancy_r']:.3f}R")
        rows.append({"variant": label, "n": m["n_trades"],
                     "per_month": round(m["n_trades"] / months, 1),
                     "wr": round(m["win_rate"], 3),
                     "pf": round(m["profit_factor"], 2),
                     "pnl": round(m["total_pnl"]),
                     "dd": round(m["max_dd_pct"], 3),
                     "exp_r": round(m["expectancy_r"], 3)})
    pd.DataFrame(rows).to_csv(save / "frontier_frequency_quality.csv",
                              index=False)

    # ---------- ۲) اعتبارسنجی C: سال‌به‌سال + تمرکز + duka ---------- #
    print("\n═══ ۲) C سال‌به‌سال (MT5) ═══")
    tC = run(fake_trend, sess).trades
    tC = tC.assign(year=pd.to_datetime(tC["entry_time"]).dt.year)
    print(tC.groupby("year")["pnl"].agg(["count", "sum"]).round(0).to_string())
    wins = tC.nlargest(5, "pnl")["pnl"].sum()
    print(f"  سهم ۵ برندهٔ برتر از سود خالص: {wins/tC.pnl.sum():.0%}")

    print("\n  تکرار روی منبع مستقل Dukascopy:")
    d15 = to_m15(duka_m1)
    d4 = resample_tf(duka_m1, "4h")
    dreg = RegimeEngine().compute(d15)["regime"].to_numpy()
    dsess = np.asarray(session_open(d15["time"]), dtype=bool)
    dfake = np.full(len(d15), TREND_UP)

    def run_duka(reg_arr):
        strat = GatedStrategy(V0Strategy(d4, rr=2.5), reg_arr, dsess,
                              direction=+1)
        bt = BacktestConfig(start_equity=30000.0, risk_pct=0.005,
                            max_positions=1, cooldown_bars=1,
                            slippage_usd=0.05, spread_gate_usd=0.40)
        pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                           pause_at=5, halt_at=6))
        return Backtester(d15, strat, bt, risk_policy=pol).run().metrics

    for label, r in (("A پایه", dreg), ("C بدون گیت رژیم", dfake)):
        m = run_duka(r)
        print(f"    duka {label:20s} {m['n_trades']:4d}t | برد {m['win_rate']:.1%} | "
              f"PF {m['profit_factor']:.2f} | ${m['total_pnl']:8,.0f} | "
              f"DD {m['max_dd_pct']:5.1%} | {m['expectancy_r']:.3f}R")

    # ---------- ۳) walk-forward دو-بازویی ---------- #
    print("\n═══ ۳) Walk-Forward دو-بازویی (IS=12ماه، OOS=1ماه) ═══")
    print("  در هر پنجره، انتخاب «گیت روشن/خاموش» فقط با ۱۲ ماه قبل از OOS")

    def run_mask(mask, reg_arr):
        bars = m15[mask].reset_index(drop=True)
        s = np.asarray(session_open(bars["time"]), dtype=bool)
        strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg_arr[mask], s,
                              direction=+1)
        res = Backtester(bars, strat,
                         BacktestConfig(start_equity=3000.0, fixed_lots=0.01,
                                        max_positions=1, cooldown_bars=1,
                                        slippage_usd=0.05,
                                        spread_gate_usd=0.40)).run()
        t = res.trades
        return (float(t["pnl"].sum()) if len(t) else 0.0), len(t), t

    keys = m15["time"].dt.strftime("%Y-%m").to_numpy()
    windows = monthly_windows(keys, 12, 1)
    print(f"  پنجره‌های OOS: {len(windows)}")

    rows, gate_f, free_f, adap_f = [], [], [], []
    for is_mask, oos_mask in windows:
        pg, _, _ = run_mask(is_mask, regime)
        pfr, nfr, _ = run_mask(is_mask, fake_trend)
        arm = "free" if (nfr >= 30 and pfr > pg) else "gated"
        _, _, t_ad = run_mask(oos_mask, fake_trend if arm == "free" else regime)
        _, _, t_g = run_mask(oos_mask, regime)
        _, _, t_f = run_mask(oos_mask, fake_trend)
        adap_f.append(t_ad); gate_f.append(t_g); free_f.append(t_f)
        rows.append({"oos_month": ",".join(sorted(set(keys[oos_mask]))),
                     "arm": arm, "is_gated_pnl": round(pg),
                     "is_free_pnl": round(pfr),
                     "oos_adaptive_pnl": float(t_ad["pnl"].sum())
                     if len(t_ad) else 0.0})
    pd.DataFrame(rows).to_csv(save / "wfo_gate_choice_per_window.csv",
                              index=False)

    dist = pd.Series([r["arm"] for r in rows]).value_counts()
    for k, v in dist.items():
        print(f"    انتخاب بازو — {k}: {v} ماه")

    def agg(frames):
        t = pd.concat(frames, ignore_index=True)
        gw = t.loc[t.pnl > 0, "pnl"].sum()
        gl = -t.loc[t.pnl <= 0, "pnl"].sum()
        return {"n": len(t), "per_month": len(t) / len(windows),
                "wr": (t.pnl > 0).mean(), "pf": gw / gl if gl else np.inf,
                "pnl": t.pnl.sum()}

    print("\n  نتایج تجمیعی OOS (fixed 0.01 lot):")
    for label, fr in (("همیشه گیت روشن (فعلی)", gate_f),
                      ("همیشه بدون گیت رژیم  ", free_f),
                      ("تطبیقی ماهانه          ", adap_f)):
        a = agg(fr)
        print(f"    {label}: {a['n']:3d}t | {a['per_month']:4.1f}/ماه | "
              f"برد {a['wr']:.1%} | PF {a['pf']:.2f} | ${a['pnl']:7,.0f}")

    print("\nجمع‌بندی: فرکانس از این مسیر قابل خریدن است (۲.۴×) اما به قیمتِ")
    print("نصف‌شدن کیفیتِ هر معامله (PF 1.43→1.21). «هم تعداد هم کیفیت» = فاز ۶.")


if __name__ == "__main__":
    main()
