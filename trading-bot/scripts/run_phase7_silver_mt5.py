"""فاز ۷ — نقره روی دادهٔ MT5 خود کاربر: اسپرد واقعی + اعتبارسنجی نهایی.

اجرا:
    py scripts/run_phase7_silver_mt5.py

سؤال‌ها (به ترتیب):
    1. توزیع اسپرد واقعی XAGUSD روی MetaQuotes-Demo چطور است؟ (۵.۷ سال، ۲م M1)
       → گیت $0.06 موقت نهایی شود یا عوض شود؟
    2. همان بک‌تست first-look، این‌بار با اسپردِ واقعیِ لحظهٔ ورود (نه فرض ثابت)
       روی پنجرهٔ ۳۶ماههٔ ثبت‌شده → لبه واقعی‌تر از PF 1.21 است؟
    3. پنجرهٔ کامل ۵.۷ ساله (۲۰۲۱→۲۰۲۶) = OOS اضافه؛ نقرهٔ ارزان ۲۰۲۱-۲۳ چه می‌گوید؟

پارامترها = همان first-look، بدون هیچ re-tuning (sl_pad $0.01093، slip $0.00109،
قرارداد ۵۰۰۰ اونس، فقط-خرید گیت‌شده). داده: data/xagusd_m1_mt5.csv.gz (خود بروکر،
تایم‌استامپ سرور؛ با infer_offset مثل طلا به UTC برمی‌گردد).
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

POINT = 0.001          # XAGUSD سه‌رقم اعشار → ۱ پوینت = $0.001
SL_PAD = 0.01093       # k = ADR نسبت نقره/طلا × 0.50 (فاز ۷ first-look)
SLIP = 0.00109
CONTRACT = 5000.0


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def spread_report(m1: pd.DataFrame) -> dict:
    """توزیع اسپرد دلاری — کل، سال‌به‌سال، ساعت‌به‌ساعت (زمان سرور)."""
    sp = m1["spread_usd"]
    out = {"overall": sp.describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict()}
    by_year = m1.groupby(m1["time"].dt.year)["spread_usd"] \
        .agg(["median", lambda s: s.quantile(0.95), "max"])
    by_year.columns = ["median", "p95", "max"]
    by_hour = m1.groupby(m1["time"].dt.hour)["spread_usd"].median()
    out["by_year"] = by_year.round(4)
    out["by_hour"] = by_hour.round(4)
    return out


def run_backtest(m15, h4, reg, sess, gate=None, start=None, end=None):
    m, r, s = m15, reg, sess
    if start is not None or end is not None:
        # نکتهٔ حیاتی: reg/sess با «موقعیت» ایندکس می‌شوند → باید با همان
        # ماسکِ پنجره برش بخورند، وگرنه رژیمِ ۲۰۲۱ به بارِ ۲۰۲۴ می‌خورد!
        mask = pd.Series(True, index=m15.index)
        t = m15["time"]
        if start is not None:
            mask &= t >= start
        if end is not None:
            mask &= t <= end
        m, r, s = m15[mask], np.asarray(reg)[mask.to_numpy()], \
            np.asarray(sess)[mask.to_numpy()]
    strat = GatedStrategy(V0Strategy(h4, rr=2.5, sl_pad=SL_PAD),
                          r, s, direction=+1)
    res = Backtester(m, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, contract_oz=CONTRACT,
        max_positions=1, cooldown_bars=1, slippage_usd=SLIP,
        spread_gate_usd=gate)).run()
    return res


def fmt(res, months) -> str:
    mm = res.metrics
    t = res.trades
    return (f"{len(t):4d}t {len(t)/months:5.1f}/ماه | برد {(t['pnl'] > 0).mean():5.1%} | "
            f"PF {mm['profit_factor']:.2f} | ${mm['total_pnl']:8,.0f} | "
            f"{mm['expectancy_r']:+.3f}R | DD {mm['max_dd_pct']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۷: نقره با اسپرد واقعی MT5")
    ap.add_argument("--mt5", default="data/xagusd_m1_mt5.csv.gz")
    ap.add_argument("--duka", default="data/xagusd_m1_duka.csv.gz")
    args = ap.parse_args()

    mt5 = load_mt5(find(pathlib.Path(args.mt5).name), point=POINT)
    duka = load_duka(find(pathlib.Path(args.duka).name))
    print(f"📊 نقره MT5: {len(mt5):,} کندل M1 | "
          f"{mt5['time'].iloc[0]:%Y-%m-%d} → {mt5['time'].iloc[-1]:%Y-%m-%d}")

    # ---------- ۱) توزیع اسپرد ----------
    rep = spread_report(mt5)
    o = rep["overall"]
    print(f"\n═══ اسپرد واقعی (۵.۷ سال) ═══")
    print(f"  میانه ${o['50%']:.3f} | میانگین ${o['mean']:.3f} | "
          f"p90 ${o['90%']:.3f} | p95 ${o['95%']:.3f} | p99 ${o['99%']:.3f} | "
          f"حداکثر ${o['max']:.2f}")
    print("\n  سال‌به‌سال (میانه | p95 | حداکثر):")
    print(rep["by_year"].to_string())
    sp = mt5["spread_usd"]
    print("\n  سهم کندل‌های بالای گیت‌های کاندید:")
    for g in (0.04, 0.06, 0.08):
        print(f"    > ${g:.2f}: {(sp > g).mean():.1%}")
    worst_h = rep["by_hour"].idxmax()
    print(f"\n  بدترین ساعت سرور (میانه): {worst_h:02d}:00 = "
          f"${rep['by_hour'].max():.3f} | بهترین: "
          f"{rep['by_hour'].idxmin():02d}:00 = ${rep['by_hour'].min():.3f}")

    # ---------- ۲) بک‌تست با اسپرد واقعی ----------
    offs = infer_offset_minutes(mt5, duka)
    mt5_utc = shift_mt5_to_utc(mt5, offs)
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)

    # پنجرهٔ ثبت‌شدهٔ first-look (۳۶ ماه، هم‌پوشان با دادهٔ دuka)
    start = pd.Timestamp("2023-09-11", tz=m15["time"].dt.tz) \
        if m15["time"].dt.tz is not None else pd.Timestamp("2023-09-11")
    end = pd.Timestamp("2026-09-09", tz=start.tz) if start.tz is not None \
        else pd.Timestamp("2026-09-09")

    print("\n═══ بک‌تست نقره — اسپرد واقعی لحظهٔ ورود ═══")
    print(f"  پنجرهٔ ۳۶ماهه (مقایسه با first-look دuka):")
    r36 = run_backtest(m15, h4, reg, sess, start=start, end=end)
    print(f"    بدون گیت:      {fmt(r36, 36.0)}")
    r36g = run_backtest(m15, h4, reg, sess, gate=0.06, start=start, end=end)
    print(f"    با گیت $0.06:  {fmt(r36g, 36.0)}")
    y = r36.trades.assign(y=pd.to_datetime(r36.trades["entry_time"]).dt.year) \
        .groupby("y")["pnl"].sum().round(0)
    print(f"    سال‌به‌سال: {y.to_dict()}")

    # پنجرهٔ کامل ۵.۷ ساله — OOS اضافه (نقرهٔ $22-26 دوران ۲۰۲۱-۲۳)
    months_full = (mt5_utc["time"].iloc[-1] - mt5_utc["time"].iloc[0]).days / 30.44
    print(f"\n  پنجرهٔ کامل {months_full:.0f} ماهه (۲۰۲۱→۲۰۲۶):")
    rfull = run_backtest(m15, h4, reg, sess)
    print(f"    بدون گیت:      {fmt(rfull, months_full)}")
    rfullg = run_backtest(m15, h4, reg, sess, gate=0.06)
    print(f"    با گیت $0.06:  {fmt(rfullg, months_full)}")
    yf = rfull.trades.assign(
        y=pd.to_datetime(rfull.trades["entry_time"]).dt.year) \
        .groupby("y")["pnl"].sum().round(0)
    print(f"    سال‌به‌سال: {yf.to_dict()}")

    try:
        out = find(pathlib.Path(args.mt5).name).parent
        pd.DataFrame([{
            "median_spread": round(o["50%"], 4),
            "p95_spread": round(o["95%"], 4),
            "pf_36mo_real": round(r36.metrics["profit_factor"], 2),
            "pnl_36mo_real": round(r36.metrics["total_pnl"], 0),
            "pf_36mo_gated": round(r36g.metrics["profit_factor"], 2),
            "pf_full": round(rfull.metrics["profit_factor"], 2),
            "pnl_full": round(rfull.metrics["total_pnl"], 0),
        }]).to_csv(out / "phase7_silver_mt5_real.csv", index=False)
        print(f"\n💾 {out}/phase7_silver_mt5_real.csv")
    except OSError as e:
        print(f"  (ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
