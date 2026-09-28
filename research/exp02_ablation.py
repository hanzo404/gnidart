"""آزمایش ۰۲ — کالبدشکافی: کدام جزء از زنجیرهٔ ICT واقعاً ارزش دارد؟

روش: هر بار یک حلقه از زنجیره را خاموش می‌کنیم و اثرش را بر امید ریاضی
(پس از هزینهٔ واقعی) می‌بینیم. هر جزء که سودِ افزوده نداشته باشد، از
استراتژی نهایی حذف می‌شود — حتی اگر در روایت ICT مرکزی باشد.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, load_m1, resample, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, halve_report, monthly_table

pd.set_option("display.width", 240)

CONFIGS = {
    "A_sweep_only":      dict(use_bias=False, require_mss=False, require_fvg=False),
    "B_sweep+disp":      dict(use_bias=False, require_mss=False, require_fvg=False, disp_atr=0.8),
    "C_sweep+mss":       dict(use_bias=False, require_mss=True,  require_fvg=False),
    "D_sweep+fvg":       dict(use_bias=False, require_mss=False, require_fvg=True),
    "E_full_no_bias":    dict(use_bias=False, require_mss=True,  require_fvg=True),
    "F_full_with_bias":  dict(use_bias=True,  require_mss=True,  require_fvg=True),
    "G_full_soft_disp":  dict(use_bias=True,  require_mss=True,  require_fvg=True, disp_atr=0.3),
    "H_full_tight_fvg":  dict(use_bias=True,  require_mss=True,  require_fvg=True, fvg_min_atr=0.30),
}


def run(symbol: str, configs: dict, rr: float = 2.0) -> pd.DataFrame:
    sp = SPECS[symbol]
    m1 = load_m1(sp)
    m5 = resample(m1, 5)
    ok = np.asarray(valid_bars(m5, sp))
    a5 = atr(m5, 14)
    sweeps = detect_sweeps(m5, StructureParams(), a5)
    sweeps = sweeps[ok[sweeps.bar.values]].reset_index(drop=True)
    print(f"\n{'='*110}\n{symbol}: M5={len(m5):,}  sweeps={len(sweeps):,}  "
          f"spread={sp.spread_pts}pts  RR={rr}")

    sim_p = SimParams(rr=rr, min_stop_spread=4.0, max_hold_bars=90)
    spread = sp.spread_pts * sp.point
    rows, all_trades = [], {}
    for cname, over in configs.items():
        p = SetupParams(**over)
        sig = build_signals(m5, sweeps, a5, p, sp.point, spread)
        tr = simulate(m1, sig, sp, sim_p)
        m = metrics(tr)
        all_trades[cname] = tr
        rows.append({"config": cname, "signals": len(sig), **{k: m.get(k) for k in
                     ("n", "win_rate", "expectancy_r", "profit_factor", "total_r",
                      "avg_win_r", "avg_loss_r", "max_dd_pct", "max_consec_loss",
                      "tp_rate", "avg_bars", "final_equity")}})
        print(f"  {cname:18s} sig={len(sig):6,}  n={m.get('n',0):5,}  "
              f"WR={m.get('win_rate',0):5.1f}%  E[R]={m.get('expectancy_r',0):+6.3f}  "
              f"PF={m.get('profit_factor',0):5.2f}  Σr={m.get('total_r',0):+7.1f}  "
              f"DD={m.get('max_dd_pct',0):6.1f}%  TP%={m.get('tp_rate',0):4.1f}")
    return pd.DataFrame(rows), all_trades


def main() -> None:
    os.makedirs("results", exist_ok=True)
    summary, keep = [], {}
    for sym in SPECS:
        df, tr = run(sym, CONFIGS)
        df.insert(0, "symbol", sym)
        summary.append(df)
        keep[sym] = tr
    out = pd.concat(summary, ignore_index=True)
    out.to_csv("results/exp02_ablation.csv", index=False)

    # تحلیل عمیق روی بهترین پیکربندی
    best = out.sort_values("expectancy_r", ascending=False).iloc[0]
    print(f"\n{'='*110}\nبهترین پیکربندی خام: {best.config} ({best.symbol})  E[R]={best.expectancy_r:+.3f}")
    for sym, _cfg_trades in keep.items():
        tr = _cfg_trades.get(best.config)
        if tr is None or tr.empty:
            continue
        print(f"\n──── {sym} / {best.config}: تقسیم نیمه‌ای (پایداری) ────")
        print(halve_report(tr).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
        mt = monthly_table(tr)
        print(f"  ماه‌ها: {len(mt)}  |  ماه‌های مثبت: {(mt.r_sum>0).sum()}  "
              f"|  بدترین ماه: {mt.r_sum.min():+.1f}R  |  بهترین: {mt.r_sum.max():+.1f}R")
        tr.to_csv(f"results/trades_{sym}_{best.config}.csv", index=False)


if __name__ == "__main__":
    main()
