"""تولید فهرست سیگنال‌های موردانتظار برای بازهٔ دلخواه.

هدف: پاریتی (انطباق) بین پایتون و MQL5 را بتوان *اثبات* کرد، نه ادعا.

روش کار:
  ۱) این اسکریپت را روی یک بازهٔ مشخص اجرا کنید و CSV سیگنال بسازید.
  ۲) همان بازه را در آزمونگر استراتژی MT5 با ربات اجرا کنید
     (LiveTrading=false ⇒ ربات سیگنال‌ها را در فایل لاگ می‌نویسد و هیچ
     سفارشی نمی‌فرستد).
  ۳) دو فایل را مقایسه کنید. اگر زمان و جهت و قیمت ورود یکی باشند،
     منطق پورت‌شده دقیقاً همان چیزی است که در پژوهش سنجیده شد.

نکته: دادهٔ M1 باید همان بازه و همان نماد باشد. تفاوت «کندل جاری» بین
بروکر و Dukascopy طبیعی است؛ معیار، توالی سیگنال‌هاست نه ثانیهٔ دقیق.
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


# مقادیر پیش‌فرض = همان‌هایی که در Esprakt_M1_LiquiditySweep.mq5 هستند
PRESET = dict(
    swing_k=3, pool_tol_atr=0.25,
    use_bias=False, require_mss=True, require_fvg=False,
    disp_atr=0.80, disp_max_bars=3, mss_lookback=10, mss_max_bars=4,
    stop_buffer_atr=0.15, min_stop_atr=0.30, max_stop_atr=3.0, min_stop_spread=3.0,
    setup_max_age=4,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US30", choices=list(SPECS))
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--out", default="results/parity_signals.csv")
    a = ap.parse_args()

    sp = SPECS[a.symbol]
    m1 = load_m1(sp)
    if a.start:
        m1 = m1[m1.time >= pd.Timestamp(a.start)]
    if a.end:
        m1 = m1[m1.time <= pd.Timestamp(a.end)]
    print(f"{a.symbol}: {len(m1):,} کندل  {m1.time.min()} → {m1.time.max()}")

    atr_ = atr(m1, PRESET["atr_period"] if "atr_period" in PRESET else 14)
    ok = np.asarray(valid_bars(m1, sp))
    sp_struct = StructureParams(swing_k=PRESET["swing_k"], pool_tol_atr=PRESET["pool_tol_atr"])
    sw = detect_sweeps(m1, sp_struct, atr_)
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
    sig = build_signals(m1, sw, atr_, params, sp.point, sp.spread_pts * sp.point)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sig.to_csv(a.out, index=False)
    print(f"سیگنال: {len(sig):,}  →  {a.out}")
    if not sig.empty:
        print(sig.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
