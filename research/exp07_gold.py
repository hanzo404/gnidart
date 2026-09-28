"""آزمایش ۰۷ — اعتبارسنجی روی دادهٔ واقعی طلا (XAUUSD).

این همان آزمونی است که در گزارش نوشته شده بود «اثبات نشده».
ابزار: ۱٬۰۶۱٬۴۹۱ کندل M1، ۲۰۲۳-۰۹ تا ۲۰۲۶-۰۹، با ستون spread واقعی
(میانه ۰٫۱۴ دلار) به‌علاوهٔ اسلیپیج تخمینی.

نکتهٔ حیاتی که از قبل پیش‌بینی شده بود:
    نسبت دامنهٔ M1 به اسپرد روی طلا ≈ ۶ برابر است، در برابر ≈ ۱۱۹ برابر
    روی شاخص‌ها. یعنی هر معامله چند برابر گران‌تر تمام می‌شود.
سؤال این است: آیا لبه این را تحمل می‌کند؟
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import Spec, load_m1, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, block_bootstrap_tstat, monthly_table, halve_report

pd.set_option("display.width", 240)

GOLD = Spec("XAUUSD", "xauusd_m1_mt5_3y.csv.gz", point=0.01, spread_pts=14.0, slippage_pts=3.0)
# slippage ۰٫۰۳ دلار در هر سمت (تخمین واقع‌بینانه برای حساب ECN)

CONFIGS = {
    "sweep_only":  dict(use_bias=False, require_mss=False, require_fvg=False),
    "sweep+mss":   dict(use_bias=False, require_mss=True,  require_fvg=False),
    "full_chain":  dict(use_bias=True,  require_mss=True,  require_fvg=True),
}
RR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]


def main() -> None:
    m1 = load_m1(GOLD)
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, GOLD))
    sw = detect_sweeps(m1, StructureParams(), a)
    if not sw.empty:
        sw = sw[ok[sw.bar.values]].reset_index(drop=True)
    spread = GOLD.spread_pts * GOLD.point
    print(f"{'='*104}\nXAUUSD  {m1.time.min().date()}→{m1.time.max().date()}  "
          f"کندل={len(m1):,}  جاروب={len(sw):,}  اسپرد میانه={spread:.3f}")

    rows, trades = [], {}
    for cname, over in CONFIGS.items():
        for rr in RR_GRID:
            sig = build_signals(m1, sw, a, SetupParams(**over), GOLD.point, spread,
                                spread_col=m1.spread.values)
            tr = simulate(m1, sig, GOLD, SimParams(rr=rr, max_hold_bars=60, min_stop_spread=3.0),
                          spread_col=m1.spread.values)
            if tr is None or len(tr) < 40:
                rows.append({"config": cname, "rr": rr, "n": 0})
                print(f"  {cname:11s} rr={rr}: سیگنال کافی نبود (n={0 if tr is None else len(tr)})")
                continue
            m, bs = metrics(tr), block_bootstrap_tstat(tr)
            rows.append({"config": cname, "rr": rr, "n": m["n"], "wr": m["win_rate"],
                         "e_r": m["expectancy_r"], "pf": m["profit_factor"],
                         "total_r": m["total_r"], "dd": m["max_dd_pct"], "t": bs["t"],
                         "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"]})
            trades[(cname, rr)] = tr
            print(f"  {cname:11s} rr={rr}: n={m['n']:5,}  WR={m['win_rate']:4.1f}%  "
                  f"E[R]={m['expectancy_r']:+.3f}  PF={m['profit_factor']:5.2f}  "
                  f"t={bs['t']:+5.2f}  CI=[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]  "
                  f"DD={m['max_dd_pct']:5.1f}%", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("results/exp07_gold.csv", index=False)

    # تحلیل عمیق روی بهترین پیکربندیِ معتبر
    valid = res[(res.n >= 150) & (res.e_r > 0) & (res.ci_lo > 0)]
    if valid.empty:
        best = res[res.n >= 150].sort_values("e_r", ascending=False).iloc[0]
    else:
        best = valid.sort_values("e_r", ascending=False).iloc[0]
    key = (best.config, best.rr)
    tr = trades.get(key)
    print(f"\n{'='*104}\nتحلیل عمیق: {key}")
    if tr is not None and not tr.empty:
        print(halve_report(tr).round(3).to_string(index=False))
        for y, g in tr.groupby(tr.time.dt.year):
            m = metrics(g)
            print(f"  {y}: n={m['n']:5,}  E[R]={m['expectancy_r']:+.3f}  "
                  f"WR={m['win_rate']:4.1f}%  PF={m['profit_factor']:.2f}")
        for d, g in tr.groupby("dir"):
            m = metrics(g)
            print(f"  {'خرید' if d==1 else 'فروش'}: n={m['n']:5,}  E[R]={m['expectancy_r']:+.3f}  "
                  f"WR={m['win_rate']:4.1f}%  PF={m['profit_factor']:.2f}")
        mt = monthly_table(tr)
        print(f"  ماهانه: {len(mt)} ماه | مثبت {(mt.r_sum>0).sum()} ({100*(mt.r_sum>0).mean():.0f}%) | "
              f"بدترین {mt.r_sum.min():+.1f}R | بهترین {mt.r_sum.max():+.1f}R")
        tr.to_csv("results/exp07_gold_trades.csv", index=False)

    # ── مقاومت در برابر هزینه: اینجا‌جاست که طلا خودش را نشان می‌دهد
    print(f"\n{'='*104}\nمقاومت در برابر هزینه (اسپرد و اسلیپیج واقعیِ داده در حالت پایه)")
    for mult in (1, 2, 3):
        g2 = Spec(GOLD.symbol, GOLD.file, GOLD.point, GOLD.spread_pts * mult,
                  GOLD.slippage_pts * mult)
        sig = build_signals(m1, sw, a, SetupParams(**CONFIGS[best.config]), g2.point,
                            g2.spread_pts * g2.point, spread_col=m1.spread.values * mult)
        t2 = simulate(m1, sig, g2, SimParams(rr=best.rr, max_hold_bars=60, min_stop_spread=3.0),
                      spread_col=m1.spread.values * mult)
        m, bs = metrics(t2), block_bootstrap_tstat(t2)
        print(f"  هزینه ×{mult}: n={m['n']:5,}  E[R]={m['expectancy_r']:+.3f}  "
              f"PF={m['profit_factor']:.2f}  t={bs['t']:+.2f}  "
              f"CI=[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
