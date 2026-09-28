"""آزمایش ۰۶ — آزمون صریحِ ادعای «کیل‌زون» و نقشهٔ ساعتی.

پرسش: آیا ساعت‌هایی که ICT «کیل‌زون» می‌نامد (۳-۴، ۱۰-۱۱، ۱۴-۱۵ به وقت
نیویورک) در دادهٔ ما هم برترند؟ و آیا «ساعت‌های کشف‌شده از داده» در نیمهٔ
دومِ دیده‌نشده تکرار می‌شوند؟

ریسکِ این آزمون: انتخاب ساعت از روی داده یعنی «شکارِ نویز». برای همین
ساعت‌ها را روی نیم��ٔ اول پیدا و روی نیمهٔ دوم می‌آزماییم.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, load_m1, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import block_bootstrap_tstat

pd.set_option("display.width", 240)
CFG = dict(use_bias=False, require_mss=True, require_fvg=False)   # پیکربندی منتخب
SIM = dict(rr=1.5, max_hold_bars=60, min_stop_spread=3.0)
ICT_WINDOWS = [(3, 4), (10, 11), (14, 15)]        # ساعت نیویورک


def run(symbol: str) -> pd.DataFrame:
    sp = SPECS[symbol]
    m1 = load_m1(sp)
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, sp))
    sw = detect_sweeps(m1, StructureParams(), a)
    sw = sw[ok[sw.bar.values]].reset_index(drop=True)
    sig = build_signals(m1, sw, a, SetupParams(**CFG), sp.point, sp.spread_pts * sp.point)
    tr = simulate(m1, sig, sp, SimParams(**SIM))
    tr.to_csv(f"results/exp06_trades_{symbol}.csv", index=False)
    del m1, sig, sw
    return tr


def main() -> None:
    for symbol in SPECS:
        tr = run(symbol)
        if tr is None or tr.empty:
            continue
        t = tr.sort_values("time").copy()
        t["hour_utc"] = t.time.dt.hour
        t["hour_ny"] = (t.hour_utc + 5) % 24
        mid = t.time.iloc[len(t) // 2]
        print(f"\n{'='*100}\n{symbol}: n={len(t):,}  E[R]={t.r.mean():+.3f}  "
              f"بازه={t.time.min().date()}→{t.time.max().date()}", flush=True)

        # ── الف) آزمون مستقیم کیل‌زونِ ICT
        inside = np.zeros(len(t), bool)
        for lo_, hi_ in ICT_WINDOWS:
            inside |= t.hour_ny.between(lo_, hi_ - 1).values
        for label, mask in (("داخل کیل‌زون ICT", inside), ("بیرون کیل‌زون", ~inside)):
            g = t[mask]
            if len(g) > 30:
                bs = block_bootstrap_tstat(g)
                print(f"   {label:18s}: n={len(g):6,} ({100*len(g)/len(t):4.1f}%)  "
                      f"E[R]={g.r.mean():+.3f}  t={bs['t']:+5.2f}  "
                      f"CI=[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]", flush=True)

        # ── ب) نقشهٔ ساعتی UTC: کشف روی نیمهٔ اول، آزمون روی نیمهٔ دوم
        is_h, oos_h = t[t.time <= mid], t[t.time > mid]
        rows = []
        for h in range(24):
            a1, a2 = is_h[is_h.hour_utc == h].r, oos_h[oos_h.hour_utc == h].r
            if len(a1) < 40:
                continue
            rows.append({"h_utc": h, "h_ny": (h + 5) % 24, "is_n": len(a1), "is_r": a1.mean(),
                         "oos_n": len(a2), "oos_r": a2.mean() if len(a2) else np.nan,
                         "all_n": len(t[t.hour_utc == h]), "all_r": t[t.hour_utc == h].r.mean()})
        p = pd.DataFrame(rows)
        p.to_csv(f"results/exp06_hourmap_{symbol}.csv", index=False)
        best = p.nlargest(4, "is_r").h_utc.tolist()
        sel = oos_h[oos_h.hour_utc.isin(best)]
        non = oos_h[~oos_h.hour_utc.isin(best)]
        print(f"   ساعت‌های برترِ نیمهٔ اول: {sorted(best)} UTC", flush=True)
        print(f"     OOS داخل ساعت‌های منتخب: n={len(sel):5,}  E[R]={sel.r.mean():+.3f}  "
              f"WR={100*(sel.r>0).mean():.1f}%", flush=True)
        print(f"     OOS بقیهٔ ساعت‌ها      : n={len(non):5,}  E[R]={non.r.mean():+.3f}  "
              f"WR={100*(non.r>0).mean():.1f}%", flush=True)
        print("   نقشهٔ ساعتی (IS → OOS):")
        print(p.sort_values("is_r", ascending=False).head(10).round(3).to_string(index=False), flush=True)
        del t, tr


if __name__ == "__main__":
    main()
