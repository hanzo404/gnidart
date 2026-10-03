"""تولید مرجع پاریتی برای مقایسهٔ پایتون ↔ MQL5.

هدف: انطباق را **اثبات** کنیم، نه ادعا.

خروجی دو فایل:
  ۱) signals  — هر سیگنالی که ربات باید بسازد (زمان، جهت، استاپ، ورود)
  ۲) trades   — هر معاملهٔ اجراشده (زمان ورود واقعی، قیمت، نتیجه، R)

روش استفاده:
  ۱) این اسکریپت را روی یک بازهٔ کوتاه اجرا کنید.
  ۲) همان بازه را در آزمونگر MT5 با ربات اجرا کنید (LiveTrading=false
     ⇒ ربات سیگنال‌ها را در فایل لاگ می‌نویسد و هیچ سفارشی نمی‌فرستد).
  ۳) دو فایل را کنار هم بگذارید و **سطر به سطر** مقایسه کنید.

مهم: دادهٔ M1 باید همان نماد و همان بازه باشد. جابه‌جایی چنددقیقه‌ای بین
بروکر و Dukascopy طبیعی است؛ معیار، **توالی و جهت و نسبتِ قیمت** است.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, load_m1, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics

# دقیقاً همان مقادیری که در Esprakt_M1_LiquiditySweep.mq5 هستند.
# اگر اینجا عددی را عوض کنید، پاریتی بی‌معنا می‌شود.
PRESET = dict(
    swing_k=3, pool_tol_atr=0.25, atr_period=14,
    use_bias=False, require_mss=True, require_fvg=False,
    disp_atr=0.80, disp_max_bars=3, mss_lookback=10, mss_max_bars=4,
    stop_buffer_atr=0.15, min_stop_atr=0.30, max_stop_atr=3.0, min_stop_spread=3.0,
    setup_max_age=4,
    rr=1.5, expiry_minutes=30, max_hold_bars=60,
)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    """خواناتر کردن خروجی برای چشم انسان."""
    d = df.copy()
    for c in ("time", "sweep_time"):
        if c in d:
            d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("entry_time", "exit_time"):
        if c in d:
            d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("entry", "stop", "extreme", "atr", "target", "exit", "result_r"):
        if c in d:
            d[c] = d[c].round(2)
    if "dir" in d:
        d["dir"] = d["dir"].map({1: "BUY", -1: "SELL"})
    if "reason" in d:   # sim رشته می‌دهد: tp / sl / time
        d["reason"] = d["reason"].astype(str).str.upper()
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US30", choices=list(SPECS))
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=60, help="حداکثر سطر چاپ‌شده")
    ap.add_argument("--outdir", default="results/parity")
    a = ap.parse_args()

    sp = SPECS[a.symbol]
    m1 = load_m1(sp)
    full_range = f"{m1.time.min().date()} → {m1.time.max().date()}"
    if a.start:
        m1 = m1[m1.time >= pd.Timestamp(a.start)]
    if a.end:
        m1 = m1[m1.time <= pd.Timestamp(a.end)]
    print(f"{a.symbol}: {len(m1):,} کندل در بازه  (کل داده: {full_range})")
    if m1.empty:
        raise SystemExit("بازه خالی است.")

    atr_ = atr(m1, PRESET["atr_period"])
    ok = np.asarray(valid_bars(m1, sp))
    sw = detect_sweeps(m1, StructureParams(swing_k=PRESET["swing_k"],
                                            pool_tol_atr=PRESET["pool_tol_atr"]), atr_)
    if not sw.empty:
        sw = sw[ok[sw.bar.values]].reset_index(drop=True)
    print(f"جاروب: {len(sw):,}")

    params = SetupParams(
        use_bias=PRESET["use_bias"], require_mss=PRESET["require_mss"],
        require_fvg=PRESET["require_fvg"], disp_atr=PRESET["disp_atr"],
        disp_max_bars=PRESET["disp_max_bars"], mss_lookback=PRESET["mss_lookback"],
        mss_max_bars=PRESET["mss_max_bars"], stop_buffer_atr=PRESET["stop_buffer_atr"],
        min_stop_atr=PRESET["min_stop_atr"], max_stop_atr=PRESET["max_stop_atr"],
        min_stop_spread=PRESET["min_stop_spread"], setup_max_age=PRESET["setup_max_age"],
    )
    spread_col = m1.spread.values if "spread" in m1.columns else None
    sig = build_signals(m1, sw, atr_, params, sp.point, sp.spread_pts * sp.point,
                        spread_col=spread_col)

    trades = None
    if not sig.empty:
        tr = simulate(m1, sig, sp,
                      SimParams(rr=PRESET["rr"], max_hold_bars=PRESET["max_hold_bars"],
                                min_stop_spread=PRESET["min_stop_spread"]),
                      spread_col=spread_col)
        trades = tr

    os.makedirs(a.outdir, exist_ok=True)
    sf = f"{a.outdir}/{a.symbol}_signals.csv"
    sig.to_csv(sf, index=False)
    print(f"\nسیگنال: {len(sig):,}  →  {sf}")
    if not sig.empty:
        show = fmt(sig.head(a.limit))
        print(show.to_string(index=False))
        if len(sig) > a.limit:
            print(f"... ({len(sig) - a.limit} سطر دیگر در فایل)")

    if trades is not None and not trades.empty:
        tf = f"{a.outdir}/{a.symbol}_trades.csv"
        trades.to_csv(tf, index=False)
        print(f"\nمعامله: {len(trades):,}  →  {tf}")
        print(fmt(trades.head(a.limit)).to_string(index=False))

        # خلاصهٔ عددی: همین اعداد را در آزمونگر MT5 ببینید
        m = metrics(trades, risk_pct=0.5)
        print("\n" + "═" * 78)
        print("  عدد مرجع — همین‌ها را در آزمونگر MT5 ببینید (با اختلاف جزئیِ طبیعی)")
        print("═" * 78)
        print(f"  تعداد معامله : {m['n']:,}")
        print(f"  نرخ برد      : {m['win_rate']:.1f}٪")
        print(f"  امید ریاضی  : {m['expectancy_r']:+.3f} R")
        print(f"  ضریب سود    : {m['profit_factor']:.2f}")
        print(f"  مجموع R     : {m['total_r']:+.2f} R")
        print(f"  افت سرمایه   : {m['max_dd_pct']:.1f}٪")
        print(f"  میانگین کندل نگهداری: {m['avg_bars']:.1f}")
        print(f"  نسبت سود/زیان: {m['payoff']:.2f}")
        print("═" * 78)
    else:
        print("\n⛔ هیچ معامله‌ای اجرا نشد — احتمالاً فیلتر فاصلهٔ استاپ همه را حذف کرده.")
        print("   این با بروکری که اسپرد خیلی باز دارد طبیعی است.")


if __name__ == "__main__":
    main()
