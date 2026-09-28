"""آزمایش ۰۹ — آیا تایم‌فریم بالاتر مشکلِ هزینهٔ طلا را حل می‌کند؟

منطق آزمایش: مشکلِ طلا این است که «مقیاسِ زمانیِ سیگنال» کوتاه‌تر از
«مقیاسِ زمانیِ هزینه» است. استاپِ ساختاری روی M1 جایی می‌افتد که فاصله‌اش
از ورود فقط ~۲ برابرِ اسپرد است.

راه‌حلِ احتمالی: روی تایم‌فریم بالاتر، کندل بزرگ‌تر می‌شود ولی اسپرد ثابت
می‌ماند ⇒ نسبتِ ساختاری بهتر می‌شود.

روش: سیگنال روی M5 و M15 ساخته می‌شود، اجرا روی M1 (تا درون‌کندلی واقعی
باشد) — دقیقاً همان کاری که برای شاخص‌ها شد.
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import Spec, load_m1, resample, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, block_bootstrap_tstat

pd.set_option("display.width", 250)

GOLD = Spec("XAUUSD", "xauusd_m1_mt5_3y.csv.gz", point=0.01, spread_pts=14.0, slippage_pts=3.0)
CONFIGS = {
    "sweep_only": dict(use_bias=False, require_mss=False, require_fvg=False),
    "sweep+mss":  dict(use_bias=False, require_mss=True,  require_fvg=False),
}
RR_GRID = [1.0, 1.5, 2.0, 3.0]
HOLDS = [60, 120, 240]


def main() -> None:
    m1 = load_m1(GOLD)
    spread = m1.spread.values
    rows = []
    for tf in (5, 15):
        bars = resample(m1, tf)
        bars["spread"] = bars.open.map(lambda _: 0)  # مقداردهی اولیه؛ واقعی از M1 می‌آید
        # اسپردِ هر کندلِ تایم‌فریم بالا = اسپردِ کندل M1 متناظر (تقریب)
        a = atr(bars, 14)
        ok = np.asarray(valid_bars(bars, GOLD))
        sw = detect_sweeps(bars, StructureParams(), a)
        if not sw.empty:
            sw = sw[ok[sw.bar.values]].reset_index(drop=True)
        # نگاشت زمان کندلِ TF به اندیس M1 برای گرفتن اسپرد محلی
        idx = m1.time.searchsorted(bars.time.values)
        idx = np.clip(idx, 0, len(m1) - 1)
        spr_tf = spread[idx]
        print(f"\n{'='*100}\nXAUUSD {tf}م‌دقیقه: {len(bars):,} کندل، {len(sw):,} جاروب، "
              f"اسپرد میانه={np.median(spr_tf):.3f}", flush=True)

        for cname, over in CONFIGS.items():
            for buf, mult in itertools.product([0.15, 0.5, 1.0], [1.0, 2.0, 3.0]):
                sig = build_signals(bars, sw, a,
                                    SetupParams(**dict(over, stop_buffer_atr=buf,
                                                       min_stop_spread=mult)),
                                    GOLD.point, GOLD.spread_pts * GOLD.point,
                                    spread_col=spr_tf)
                for rr, hold in itertools.product(RR_GRID, HOLDS):
                    tr = simulate(m1, sig, GOLD,
                                  SimParams(rr=rr, max_hold_bars=hold, min_stop_spread=mult),
                                  spread_col=spread)
                    if tr is None or len(tr) < 120:
                        rows.append({"tf": tf, "config": cname, "buf": buf, "mult": mult,
                                     "rr": rr, "hold": hold, "n": 0 if tr is None else len(tr)})
                        continue
                    m, bs = metrics(tr), block_bootstrap_tstat(tr)
                    rows.append({"tf": tf, "config": cname, "buf": buf, "mult": mult, "rr": rr,
                                 "hold": hold, "n": m["n"], "wr": m["win_rate"],
                                 "e_r": m["expectancy_r"], "pf": m["profit_factor"],
                                 "dd": m["max_dd_pct"], "t": bs["t"],
                                 "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"]})
            print(f"  {cname} تمام شد", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("results/exp09_gold_tf.csv", index=False)
    if "ci_lo" not in res.columns:      # هیچ خانه‌ای به آستانهٔ نمونه نرسید
        res["ci_lo"] = np.nan
        res["ci_hi"] = np.nan
    ok = res[(res.n >= 200) & (res.ci_lo > 0)]
    print(f"  بیشترین تعداد معامله در کل شبکه: {int(res.n.max())}")
    print(f"\n{'='*100}\nنتیجه: {len(res)} خانه آزموده شد؛ {len(ok)} خانه معنادار (n≥۲۰۰ و CI>0)")
    if not ok.empty:
        print(ok.sort_values("e_r", ascending=False).head(20).round(3).to_string(index=False))
    else:
        print("\n  ⛔ هیچ خانه‌ای معنادار نشد.")
    for tf, g in ok.groupby("tf"):
        print(f"\n  ── بهترین‌های {tf}م‌دقیقه ──")
        print(g.sort_values("e_r", ascending=False).head(8).round(3).to_string(index=False))

    # بهترینِ هر تایم‌فریم با آزمون هزینه
    for tf, g in ok.groupby("tf"):
        if g.empty:
            continue
        b = g.sort_values("e_r", ascending=False).iloc[0]
        print(f"\n  ── دوامِ بهترینِ {tf}م‌دقیقه در برابر هزینه ──")
        for mc in (1, 2, 3):
            bars = resample(m1, tf)
            a = atr(bars, 14)
            sw = detect_sweeps(bars, StructureParams(), a)
            idx = np.clip(m1.time.searchsorted(bars.time.values), 0, len(m1) - 1)
            spr_tf = spread[idx]
            sig = build_signals(bars, sw, a,
                                SetupParams(**dict(CONFIGS[b.config], stop_buffer_atr=b.buf,
                                                   min_stop_spread=b.mult)),
                                GOLD.point, GOLD.spread_pts * GOLD.point,
                                spread_col=spr_tf * mc)
            t2 = simulate(m1, sig, GOLD, SimParams(rr=b.rr, max_hold_bars=b.hold,
                                                    min_stop_spread=b.mult),
                          spread_col=spread * mc)
            m2 = metrics(t2)
            bs2 = block_bootstrap_tstat(t2)
            print(f"    هزینه ×{mc}: n={m2['n']:5,}  E[R]={m2['expectancy_r']:+.3f}  "
                  f"PF={m2['profit_factor']:.2f}  t={bs2['t']:+.2f}  "
                  f"CI=[{bs2['ci_lo']:+.3f},{bs2['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
