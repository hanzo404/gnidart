"""آزمایش ۰۵ — تصمیم نهایی: کدام پیکربندی، و آیا «کیل‌زون» واقعاً کار می‌کند؟

سه پرسش:
    ۱) روی M1، کدام حلقه از زنجیرهٔ ICT واقعاً ارزش افزوده دارد؟
    ۲) نقشهٔ ساعتی: ساعت‌های برتر را روی نیم��ٔ اول پیدا می‌کنیم و
       روی نیمهٔ دوم *می‌آزماییم*. اگر ساعت‌های ICT (کیل‌زون) از دلِ داده
       بیرون بیایند، شاهدِ مستقلی برای روایت ICT خواهیم داشت.
    ۳) تکرار روی دو نمادِ دیده‌نشده با اسپرد واقعی.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, Spec, load_m1, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, block_bootstrap_tstat

pd.set_option("display.width", 240)

CONFIGS = {
    "sweep_only": dict(use_bias=False, require_mss=False, require_fvg=False),
    "sweep+mss":  dict(use_bias=False, require_mss=True,  require_fvg=False),
    "full_chain": dict(use_bias=True,  require_mss=True,  require_fvg=True),
}
HELDOUT = {
    "USTEC": Spec("USTEC", "ustec_m1.csv.gz", point=0.01, spread_pts=None, slippage_pts=1.0),
    "USTECH100": Spec("USTECH100", "ustech100m_m1.csv.gz", point=0.01, spread_pts=None, slippage_pts=1.0),
}


def prep(sp: Spec):
    m1 = load_m1(sp)
    if sp.spread_pts is None:
        sp = Spec(sp.symbol, sp.file, sp.point, float(m1.spread.median() * sp.point), sp.slippage_pts)
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, sp))
    sw = detect_sweeps(m1, StructureParams(), a)
    if not sw.empty:
        sw = sw[ok[sw.bar.values]].reset_index(drop=True)
    return m1, sp, a, sw


def main() -> None:
    os.makedirs("results", exist_ok=True)
    rows, saved = [], {}

    for symbol, sp0 in list(SPECS.items()) + list(HELDOUT.items()):
        m1, sp, a, sw = prep(sp0)
        spread = sp.spread_pts * sp.point
        rng_ratio = spread / (m1.high - m1.low).median()
        print(f"\n{'='*100}\n{symbol}: {m1.time.min().date()}→{m1.time.max().date()}  "
              f"spread={spread:.3f}  spread/median_range={rng_ratio:.2f}  جاروب={len(sw):,}")
        for cname, over in CONFIGS.items():
            for rr in (1.5, 2.0):
                sig = build_signals(m1, sw, a, SetupParams(**over), sp.point, spread)
                tr = simulate(m1, sig, sp, SimParams(rr=rr, max_hold_bars=60, min_stop_spread=3.0))
                if tr is None or tr.empty:
                    continue
                m, bs = metrics(tr), block_bootstrap_tstat(tr)
                rows.append({"symbol": symbol, "config": cname, "rr": rr, **m, "t": bs["t"],
                             "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"]})
                if cname == "full_chain" and rr == 2.0:
                    saved[symbol] = (m1, sp, a, sw, tr)
        for r in rows[-3:]:
            print(f"  {r['config']:10s} rr={r['rr']}: n={r['n']:6,}  WR={r['win_rate']:4.1f}%  "
                  f"E[R]={r['expectancy_r']:+.3f}  PF={r['profit_factor']:.2f}  t={r['t']:+6.2f}  "
                  f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  DD={r['max_dd_pct']:5.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv("results/exp05_configs.csv", index=False)
    print(f"\n{'='*100}\nجدول کامل:")
    print(res[["symbol","config","rr","n","win_rate","expectancy_r","profit_factor","t","ci_lo","ci_hi","max_dd_pct"]]
          .round(3).to_string(index=False))

    # ── ۲) نقشهٔ ساعتی با آزمون خارج از نمونه
    print(f"\n{'='*100}\n۲) کیل‌زون: ساعت‌های برترِ نیمهٔ اول، آزمون در نیمهٔ دوم")
    for symbol, (m1, sp, a, sw, tr) in saved.items():
        if symbol not in SPECS or tr.empty:
            continue
        t = tr.sort_values("time").copy()
        t["hour"] = t.time.dt.hour
        mid = t.time.iloc[len(t) // 2]
        is_h, oos_h = t[t.time <= mid], t[t.time > mid]
        prof = []
        for h in range(24):
            a1 = is_h[is_h.hour == h].r
            a2 = oos_h[oos_h.hour == h].r
            if len(a1) < 30:
                continue
            prof.append({"hour_utc": h, "ny_hour": (h - 13) % 24, "is_n": len(a1), "is_r": a1.mean(),
                         "oos_n": len(a2), "oos_r": a2.mean() if len(a2) else np.nan,
                         "oos_all_n": len(t[t.hour == h]), "all_r": t[t.hour == h].r.mean()})
        p = pd.DataFrame(prof)
        if p.empty:
            continue
        best = p.nlargest(4, "is_r").hour_utc.tolist()
        # ۲ الف) آیا ساعت‌های انتخاب‌شده در نیمهٔ دوم هم مثبت‌اند؟
        sel = oos_h[oos_h.hour.isin(best)]
        non = oos_h[~oos_h.hour.isin(best)]
        print(f"\n  ── {symbol}: ساعت‌های منتخبِ نیم��ٔ اول = {sorted(best)} (UTC)")
        print(f"     OOS ساعت‌های منتخب : n={len(sel):5,}  E[R]={sel.r.mean():+.3f}  WR={100*(sel.r>0).mean():.1f}%")
        print(f"     OOS ساعت‌های بقیه   : n={len(non):5,}  E[R]={non.r.mean():+.3f}  WR={100*(non.r>0).mean():.1f}%")
        print("     نقشهٔ ساعتی (IS → OOS):")
        print(p.sort_values("is_r", ascending=False).head(8).to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        p.to_csv(f"results/exp05_hour_{symbol}.csv", index=False)


if __name__ == "__main__":
    main()
