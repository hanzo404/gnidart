"""آزمایش ۰۸ — آیا این استراتژی روی طلا اصلاً قابل اجراست؟

یافتهٔ آزمایش ۰۷: فیلتر «فاصلهٔ استاپ ≥ k×اسپرد» روی طلا ۹۷.۶٪ سیگنال‌ها
را حذف می‌کند. علت ساختاری است:

    شاخص‌ها: دامنهٔ M1 ≈ ۱۱۹ برابرِ اسپرد  ⇒ استاپِ ساختاری راحت جا می‌شود
    طلا   : دامنهٔ M1 ≈   ۶ برابرِ اسپرد  ⇒ استاپِ ساختاری از پس‌اندازش بیرون است

پس سؤال درست این نیست که «آیا کلیکِ استراتژی روی طلا سود می‌دهد؟» (تا حدی
می‌دهد) بلکه این است که «آیا می‌توان آن را طوری تنظیم کرد که هزینه را تحمل
کند؟»

شبکهٔ آزمون: k×اسپرد (فاصلهٔ حداقلی استاپ) × بافرِ استاپ (ATR) × RR
معیار تصمیم: نه بهترین خانه، بلکه «سکوی پهنِ سودده».
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import Spec, load_m1, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, block_bootstrap_tstat

pd.set_option("display.width", 250)

GOLD = Spec("XAUUSD", "xauusd_m1_mt5_3y.csv.gz", point=0.01, spread_pts=14.0, slippage_pts=3.0)
SPREAD_MULT = [0.0, 1.0, 2.0, 3.0]      # ضریبِ اسپرد برای حداقل فاصلهٔ استاپ
BUFFER_ATR = [0.15, 0.5, 1.0, 2.0]     # بافر پشت extremum جاروب
RR_GRID = [1.0, 1.5, 2.0, 3.0]
CONFIGS = {
    "sweep_only": dict(use_bias=False, require_mss=False, require_fvg=False),
    "sweep+mss":  dict(use_bias=False, require_mss=True,  require_fvg=False),
}


def main() -> None:
    m1 = load_m1(GOLD)
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, GOLD))
    sw = detect_sweeps(m1, StructureParams(), a)
    sw = sw[ok[sw.bar.values]].reset_index(drop=True)
    spread = m1.spread.values
    print(f"XAUUSD: {len(m1):,} کندل، {len(sw):,} جاروب، اسپرد میانه={np.median(spread):.3f}")

    rows = []
    for cname, over in CONFIGS.items():
        for buf in BUFFER_ATR:
            base = dict(over, stop_buffer_atr=buf)
            for mult in SPREAD_MULT:
                # ساختن سیگنال برای این بافر، سپس تغییر آستانهٔ اسپرد
                sig = build_signals(m1, sw, a, SetupParams(**dict(base, min_stop_spread=mult)),
                                    GOLD.point, GOLD.spread_pts * GOLD.point,
                                    spread_col=spread)
                for rr in RR_GRID:
                    tr = simulate(m1, sig, GOLD,
                                  SimParams(rr=rr, max_hold_bars=60, min_stop_spread=mult),
                                  spread_col=spread)
                    if tr is None or len(tr) < 120:
                        rows.append({"config": cname, "buf": buf, "mult": mult, "rr": rr,
                                     "n": 0 if tr is None else len(tr)})
                        continue
                    m, bs = metrics(tr), block_bootstrap_tstat(tr)
                    rows.append({"config": cname, "buf": buf, "mult": mult, "rr": rr,
                                 "n": m["n"], "wr": m["win_rate"], "e_r": m["expectancy_r"],
                                 "pf": m["profit_factor"], "dd": m["max_dd_pct"],
                                 "t": bs["t"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                                 "total_r": m["total_r"]})
        print(f"  {cname} تمام شد", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("results/exp08_gold_grid.csv", index=False)
    ok = res[res.n >= 200]
    print(f"\n{'='*110}\nخلاصهٔ شبکه ({len(res)} خانه، {len(ok)} خانه با n≥200)")
    if ok.empty:
        print("  هیچ خانه‌ای به ۲۰۰ معامله نرسید ⇒ این تنظیم روی طلا اجرا نمی‌شود.")
    else:
        good = ok[(ok.ci_lo > 0) & (ok.e_r > 0.08)]
        print(f"  خانه‌های معنادار (CI>0 و E[R]>0.08): {len(good)}")
        if not good.empty:
            print(good.sort_values("e_r", ascending=False).head(20).round(3).to_string(index=False))
        print("\n  بهترین ۱۵ خانه به تفکیک پیکربندی:")
        for cname, g in ok.groupby("config"):
            print(f"\n  ── {cname} ──")
            print(g.nlargest(15, "e_r").round(3).to_string(index=False))
        # نقشهٔ سکو: میانگین E[R] روی RR برای هر (buf, mult)
        print("\n  ── سکوی سوددهی (میانگین E[R] روی همهٔ RR، n≥200) ──")
        for cname, g in ok.groupby("config"):
            piv = g.pivot_table(index="buf", columns="mult", values="e_r", aggfunc="mean")
            nn = g.pivot_table(index="buf", columns="mult", values="n", aggfunc="mean")
            print(f"  {cname}:")
            print(piv.round(3).to_string())
            print("  تعداد معاملهٔ میانگین:")
            print(nn.round(0).to_string())
        # آیا معناداری در برابر هزینهٔ ۲برابر دوام می‌آورد؟
        print("\n  ── دوام در برابر هزینه (بهترین ۵ خانهٔ معنادار) ──")
        for _, r in good.nlargest(5, "e_r").iterrows():
            out = []
            for mult_cost in (1, 2, 3):
                g2 = Spec(GOLD.symbol, GOLD.file, GOLD.point, GOLD.spread_pts * mult_cost,
                          GOLD.slippage_pts * mult_cost)
                sig = build_signals(m1, sw, a,
                                    SetupParams(**dict(CONFIGS[r.config], stop_buffer_atr=r.buf,
                                                       min_stop_spread=r.mult)),
                                    g2.point, g2.spread_pts * g2.point, spread_col=spread * mult_cost)
                t2 = simulate(m1, sig, g2, SimParams(rr=r.rr, max_hold_bars=60,
                                                      min_stop_spread=r.mult),
                              spread_col=spread * mult_cost)
                m2 = metrics(t2)
                out.append(f"×{mult_cost}: E[R]={m2['expectancy_r']:+.3f} n={m2['n']}")
            print(f"  {r.config:11s} buf={r.buf} mult={r.mult} rr={r.rr}:  " + "  |  ".join(out))


if __name__ == "__main__":
    main()
