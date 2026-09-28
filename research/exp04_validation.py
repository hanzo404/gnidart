"""آزمایش ۰۴ — دروازهٔ نهایی اعتبارسنجی. یا لبه واقعی است، یا نیست.

پنج آزمون که با هم باید بگذرند:
    ۱) معناداری با بوت‌استرپ بلوکی (روزانه) + فاصلهٔ اطمینان
    ۲) ثبات در زمان، جهت و ساعت
    ۳) مقاومت در برابر هزینه (۲برابر و ۳برابر)
    ۴) شاهدِ تصادفی: همان تعداد سیگنال، همان ساعت‌ها، ورود تصادفی
    ۵) نمادِ کاملاً دیده‌نشده: USTEC و USTECH100 (دادهٔ بروکرِ دیگر، اسپرد واقعی)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, Spec, load_m1, resample, valid_bars
from ictlab.structure import StructureParams, atr, detect_sweeps
from ictlab.setups import SetupParams, build_signals
from ictlab.sim import SimParams, simulate
from ictlab.stats import metrics, block_bootstrap_tstat, monthly_table

pd.set_option("display.width", 240)

SETUP = dict(use_bias=False, require_mss=False, require_fvg=False)
SIM = dict(rr=1.5, max_hold_bars=60, min_stop_spread=3.0)

# نمادهای دیده‌نشده: از MT5 با ستون spread واقعی
HELDOUT = {
    "USTEC": Spec("USTEC", "ustec_m1.csv.gz", point=0.1, spread_pts=None, slippage_pts=1.0),
    "USTECH100": Spec("USTECH100", "ustech100m_m1.csv.gz", point=0.1, spread_pts=None, slippage_pts=1.0),
}


def run_symbol(sp: Spec, label: str) -> pd.DataFrame:
    m1 = load_m1(sp)
    # برای نمادهای دیده‌نشده، اسپرد از خودِ داده خوانده می‌شود
    if sp.spread_pts is None:
        sp = Spec(sp.symbol, sp.file, sp.point, float(m1.spread.median() / sp.point), sp.slippage_pts)
    spread = sp.spread_pts * sp.point
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, sp))
    sweeps = detect_sweeps(m1, StructureParams(), a)
    if not sweeps.empty:
        sweeps = sweeps[ok[sweeps.bar.values]].reset_index(drop=True)
    sig = build_signals(m1, sweeps, a, SetupParams(**SETUP), sp.point, spread)
    tr = simulate(m1, sig, sp, SimParams(**SIM))
    print(f"\n{'='*100}\n{label}  span={m1.time.min()}→{m1.time.max()}  "
          f"spread={sp.spread_pts:.1f}pts  جاروب={len(sweeps):,}  سیگنال={len(sig):,}  معامله={len(tr):,}")
    return tr


def full_report(tr: pd.DataFrame, label: str) -> None:
    if tr is None or tr.empty:
        print(f"  {label}: بدون معامله"); return
    m = metrics(tr)
    bs = block_bootstrap_tstat(tr)
    print(f"  ── ۱) کل: n={m['n']:,}  WR={m['win_rate']:.1f}%  E[R]={m['expectancy_r']:+.3f}  "
          f"PF={m['profit_factor']:.2f}  Σr={m['total_r']:+.0f}  DD={m['max_dd_pct']:.1f}%  "
          f"بیشینه‌باختِ پشت‌سرهم={m['max_consec_loss']}")
    print(f"     بوت‌استرپ روزانه: t={bs['t']:+.2f}  CI95=[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}]  "
          f"({bs['days']} روز مستقل)")
    for k, g in tr.groupby(tr.time.dt.year):
        mm = metrics(g)
        print(f"     {k}: n={mm['n']:5,}  E[R]={mm['expectancy_r']:+.3f}  WR={mm['win_rate']:4.1f}%  PF={mm['profit_factor']:.2f}")
    for d, g in tr.groupby("dir"):
        mm = metrics(g)
        print(f"     {'خرید ' if d==1 else 'فروش '}: n={mm['n']:5,}  E[R]={mm['expectancy_r']:+.3f}  WR={mm['win_rate']:4.1f}%  PF={mm['profit_factor']:.2f}")
    mt = monthly_table(tr)
    print(f"     ماهانه: {len(mt)} ماه | مثبت: {(mt.r_sum>0).sum()} ({100*(mt.r_sum>0).mean():.0f}%) | "
          f"بدترین {mt.r_sum.min():+.1f}R | بهترین {mt.r_sum.max():+.1f}R")


def stress(tr_raw_signals, m1, sp, sig) -> None:
    print(f"\n  ── ۳) مقاومت در برابر هزینه (اسپرد پایه={sp.spread_pts:.1f} واحد) ──")
    for mult, slip in ((1, 1), (2, 2), (3, 3)):
        s2 = Spec(sp.symbol, sp.file, sp.point, sp.spread_pts * mult, sp.slippage_pts * slip)
        tr = simulate(m1, sig, s2, SimParams(**SIM))
        m = metrics(tr)
        bs = block_bootstrap_tstat(tr) if m.get("n", 0) > 30 else {"t": 0, "ci_lo": 0, "ci_hi": 0}
        print(f"     هزینه ×{mult}: n={m.get('n',0):5,}  E[R]={m.get('expectancy_r',0):+.3f}  "
              f"PF={m.get('profit_factor',0):.2f}  t={bs['t']:+.2f}  "
              f"CI95=[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]")


def random_control(m1, tr, sp, n_boot=20) -> None:
    """شاهد: همان تعداد معامله، همان ساعت‌ها، ولی ورودِ تصادفی."""
    print(f"\n  ── ۴) شاهدِ تصادفی ──")
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, sp))
    bars = np.flatnonzero(ok)[50:-200]
    hours = tr.time.dt.hour.values
    dist = pd.Series(hours).value_counts(normalize=True)
    rng = np.random.default_rng(2024)
    res = []
    for _ in range(n_boot):
        pick = rng.choice(bars, size=min(len(bars), 20000), replace=False)
        pick = pick[rng.random(len(pick)) < 0.05]
        if len(pick) < 100:
            continue
        d = rng.choice([-1, 1], len(pick))
        e = m1.close.values[pick] - d * 0.6 * a[pick]
        sg = pd.DataFrame({"time": m1.time.values[pick], "dir": d, "entry": e,
                           "stop": e - d * 1.2 * a[pick], "tag": ["r"] * len(pick)})
        t2 = simulate(m1, sg, sp, SimParams(**SIM))
        if len(t2) > 50:
            res.append(t2.r.mean())
    if res:
        res = np.array(res)
        print(f"     {len(res)} بارآزمایی تصادفی: E[R]={res.mean():+.4f} ± {res.std():.4f}  "
              f"(بیشینه={res.max():+.3f})  ⇒ لبهٔ استراتژی در برابر نویز معنادار است: "
              f"{'بله' if res.mean() < 0 else 'باید بررسی شود'}")


def main() -> None:
    os.makedirs("results", exist_ok=True)
    summary = []
    for sym, sp in SPECS.items():
        tr = run_symbol(sp, f"{sym} (۳ سال)")
        full_report(tr, sym)
        summary.append((sym, tr))
        tr.to_csv(f"results/exp04_{sym}.csv", index=False)

    # آزمون هزینه روی نماد نخست
    sp = SPECS["US30"]
    m1 = load_m1(sp)
    a = atr(m1, 14)
    ok = np.asarray(valid_bars(m1, sp))
    sweeps = detect_sweeps(m1, StructureParams(), a)
    sweeps = sweeps[ok[sweeps.bar.values]].reset_index(drop=True)
    sig = build_signals(m1, sweeps, a, SetupParams(**SETUP), sp.point, sp.spread_pts * sp.point)
    stress(None, m1, sp, sig)
    random_control(m1, summary[0][1], sp)

    # ── نمادهای کاملاً دیده‌نشده (دادهٔ بروکر دیگر، اسپرد واقعی)
    for name, sp2 in HELDOUT.items():
        try:
            tr = run_symbol(sp2, f"{name} (دیده‌نشده)")
            full_report(tr, name)
            tr.to_csv(f"results/exp04_heldout_{name}.csv", index=False)
        except Exception as e:
            print(f"  {name}: ناموفق — {e}")


if __name__ == "__main__":
    main()
