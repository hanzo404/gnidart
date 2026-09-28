"""آزمایش ۰۳ — مطالعهٔ پایداری: سطحِ پارامترها به‌جای «بهترین تنظیم».

درسِ کتابخانه (Evidence-Based TA / Algorithmic Trading): اگر فقط یک نقطهٔ
پارامتری سود بدهد، آن سود «نویزِ انتخاب» است. اگر یک *فلات* پهن سود بدهد،
شاید لبهٔ واقعی باشد.

روش:
    ۱) شبکه‌ای از (RR × مدت‌نگهداری × شیوهٔ ورود × پیکربندی × تایم‌فریم)
    ۲) هر خانه با هزینهٔ واقعی شبیه‌سازی می‌شود
    ۳) امید ریاضی روی «نیمهٔ دوم» (خارج از نمونه) گزارش می‌شود
    ۴) معیار تصمیم: **سهم خانه‌های سودده**، نه بهترین خانه
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, load_m1, resample, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics

pd.set_option("display.width", 240)

RR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
HOLD_GRID = [30, 60, 120, 240]
CONFIGS = {
    "sweep_only":  dict(use_bias=False, require_mss=False, require_fvg=False),
    "full_chain":  dict(use_bias=True,  require_mss=True,  require_fvg=True),
    "sweep+mss":   dict(use_bias=False, require_mss=True,  require_fvg=False),
}


def build_all(symbol: str) -> dict:
    sp = SPECS[symbol]
    m1 = load_m1(sp)
    spread = sp.spread_pts * sp.point
    out = {}
    for tf_name, tf in (("M1", 1), ("M5", 5)):
        bars = resample(m1, tf)
        ok = np.asarray(valid_bars(bars, sp))
        a = atr(bars, 14)
        sweeps = detect_sweeps(bars, StructureParams(), a)
        if not sweeps.empty:
            sweeps = sweeps[ok[sweeps.bar.values]].reset_index(drop=True)
        for cname, over in CONFIGS.items():
            sig = build_signals(bars, sweeps, a, SetupParams(**over), sp.point, spread)
            out[(tf_name, cname)] = sig
        out[("_meta", tf_name)] = (len(bars), len(sweeps))
        print(f"  {symbol} {tf_name}: {len(bars):,} کندل، {len(sweeps):,} جاروب")
    return out, m1, sp


def main() -> None:
    os.makedirs("results", exist_ok=True)
    rows = []
    for symbol in SPECS:
        print(f"\n{'='*100}\n{symbol}")
        sigs, m1, sp = build_all(symbol)
        spread = sp.spread_pts * sp.point
        for (tf_name, cname), sig in sigs.items():
            if tf_name.startswith("_") or sig.empty:
                continue
            for rr, hold in itertools.product(RR_GRID, HOLD_GRID):
                sim_p = SimParams(rr=rr, max_hold_bars=hold, min_stop_spread=3.0)
                tr = simulate(m1, sig, sp, sim_p)
                if tr is None or len(tr) < 60:
                    continue
                tr = tr.sort_values("time")
                mid = tr.time.iloc[len(tr) // 2]
                is_m, oos_m = metrics(tr[tr.time <= mid]), metrics(tr[tr.time > mid])
                rows.append({
                    "symbol": symbol, "tf": tf_name, "config": cname, "rr": rr, "hold": hold,
                    "n_all": is_m["n"] + oos_m["n"],
                    "is_r": is_m["expectancy_r"], "oos_r": oos_m["expectancy_r"],
                    "oos_wr": oos_m["win_rate"], "oos_pf": oos_m["profit_factor"],
                    "oos_n": oos_m["n"], "oos_total_r": oos_m["total_r"],
                    "oos_dd": oos_m["max_dd_pct"],
                })
    res = pd.DataFrame(rows)
    res.to_csv("results/exp03_surface.csv", index=False)

    print(f"\n{'='*100}\nخلاصهٔ سطحِ پایداری (خارج از نمونه):")
    for (sym, tf, cfg), g in res.groupby(["symbol", "tf", "config"]):
        pos = (g.oos_r > 0).mean() * 100
        both = ((g.is_r > 0) & (g.oos_r > 0)).mean() * 100
        print(f"  {sym:7s} {tf:3s} {cfg:10s}  خانه‌های OOS سودده: {pos:5.1f}%  "
              f"(هم در IS و OOS: {both:5.1f}%)  |  بهترین OOS: {g.oos_r.max():+.3f}R "
              f"(n={int(g.loc[g.oos_r.idxmax(),'oos_n'])})  |  بدترین: {g.oos_r.min():+.3f}R")

    # نمای فلات برای پیکربندی‌های اصلی روی M5
    for cfg in ("sweep_only", "full_chain"):
        g = res[(res.config == cfg) & (res.tf == "M5")]
        if g.empty:
            continue
        for sym, gg in g.groupby("symbol"):
            piv = gg.pivot_table(index="hold", columns="rr", values="oos_r")
            print(f"\n  ── سطح OOS: {sym} / M5 / {cfg} (سطر=مدت نگهداری، ستون=RR) ──")
            print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
