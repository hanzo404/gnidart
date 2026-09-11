"""فاز ۶-ب — سایزینگ احتمالاتی (متا-لیبلینگ نسخهٔ ۲).

اجرا:
    py scripts/run_phase6b_sizing.py

سؤال آزمون: آیا «وزن‌دادن» به ۱۹ کاندیدای ماهانه (به‌جای فیلترکردنِ
شکست‌خورده‌ی ۶-۱) می‌تواند هم‌زمان فرکانس و کیفیت را بالاتر از G6 ببرد؟

میلهٔ از پیش ثبت‌شده (قبل از اجرا — همان‌طور که در طرح نوشته شد):
    MT5  OOS: PF وزنی ≥ 1.4 با فرکانس ≥ ۱۳/ماه
    duka OOS: PF وزنی ≥ 1.3 و بهتر از take_all
    هر دو رد شوند → فاز دفن می‌شود و G6 ساده می‌ماند.
    (اوراکل فقط سقفِ تئوریک است — جزو داوری نیست.)

خروجی: marketdata/phase6b_results.csv + phase6b_windows.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.meta import (FEATURES, MetaWFOConfig, agg, build_dataset)
from bot.analysis.meta_sizing import (agg_weighted, run_meta_sizing_wfo)
from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, TREND_UP, session_open


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def candidates_and_ref(m15, h4, regime, sess):
    """استخر کاندیدا (گیت رژیم خنثی) + مرجع گیت کامل — همان ۶-۱."""
    fake = np.full(len(m15), TREND_UP)

    def run(reg_arr):
        strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg_arr, sess,
                              direction=+1)
        return Backtester(m15, strat, BacktestConfig(
            start_equity=3000.0, fixed_lots=0.01, max_positions=1,
            cooldown_bars=1, slippage_usd=0.05,
            spread_gate_usd=0.40)).run()

    return run(fake), run(regime)


def pipeline(m15, h4, regime, sess, label: str, cfg: MetaWFOConfig,
             rows: list, save: pathlib.Path):
    print(f"\n═══ {label} ═══")
    months = (m15["time"].iloc[-1] - m15["time"].iloc[0]).days / 30.44
    cand, ref = candidates_and_ref(m15, h4, regime, sess)
    print(f"  کاندیداها: {len(cand.trades)} ({len(cand.trades)/months:.1f}/ماه) | "
          f"برد {(cand.trades['pnl'] > 0).mean():.1%} | "
          f"PF {cand.metrics['profit_factor']:.2f}")
    print(f"  گیت کامل:  {len(ref.trades)} ({len(ref.trades)/months:.1f}/ماه) | "
          f"PF {ref.metrics['profit_factor']:.2f}")

    ds = build_dataset(m15, h4, regime, sess, cand.trades)
    res = run_meta_sizing_wfo(ds, cfg)
    per = res.per_window
    oos = res.oos
    n_win = len(per)
    print(f"  پنجره‌های OOS: {n_win} | مدل‌های انتخابی: "
          f"{per['model'].value_counts().to_dict()}")

    # مرجع گیت کامل فقط روی ماه‌های OOS
    oos_months = set()
    for lbl in per["oos_month"]:
        oos_months.update(lbl.split(","))
    ref_oos = ref.trades.assign(
        m=pd.to_datetime(ref.trades["entry_time"]).dt.strftime("%Y-%m"))
    ref_oos = ref_oos[ref_oos["m"].isin(oos_months)]

    print(f"\n  ── مقایسهٔ OOS (لات ثابت 0.01، {n_win} ماه) ──")
    arms = []
    a = agg(ref_oos, n_win)
    arms.append({"dataset": label, "arm": "گیت کامل G6", **a,
                 "exp_r": round(float(ref_oos["r"].mean()), 3) if len(ref_oos) else None,
                 "mean_m": 1.0, "dd_r": None})
    a = agg(oos, n_win)
    arms.append({"dataset": label, "arm": "take_all (بدون وزن)", **a,
                 "exp_r": round(float(oos["r"].mean()), 3),
                 "mean_m": 1.0, "dd_r": None})
    for mcol, name in (("m_stepped", "وزن پله‌ای"),
                       ("m_linear", "وزن خطی"),
                       ("m_kelly", "کلی کوچک"),
                       ("m_oracle", "اوراکل (سقف!)")):
        arms.append({"dataset": label, "arm": name,
                     **agg_weighted(oos, mcol, n_win)})

    for a in arms:
        print(f"  {a['arm']:20s} {a['n']:4d}t | {a['per_month']:5.1f}/ماه | "
              f"برد {a['wr']:.1%} | PF {a['pf']:.2f} | ${a['pnl']:7,.0f} | "
              f"exp {a['exp_r']:+.3f}R | m̄ {a['mean_m']:.2f}"
              + (f" | DD {a['dd_r']:.1f}R" if a["dd_r"] is not None else ""))
    rows.extend([{**a, "dd_r": a.get("dd_r")} for a in arms])

    # ---------- تشخیص صداقت: مدل اصلاً چیزی می‌داند؟ ----------
    print("\n  ── تشخیص: قدرت پیش‌بینی OOS ──")
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(oos["win"], oos["p"])
        print(f"  AUC(احتمال, برد) روی OOS = {auc:.3f}  (0.5 = بی‌خبری)")
    except ValueError:
        print("  AUC: قابل‌محاسبه نیست (احتمال ثابت)")
    print("  کالیبراسیون (باند پله‌ای):")
    for lo, hi, name in ((0.0, 0.35, "p<0.35"), (0.35, 0.45, "0.35–0.45"),
                         (0.45, 1.01, "p≥0.45")):
        sub = oos[(oos["p"] >= lo) & (oos["p"] < hi)]
        if len(sub):
            print(f"    {name:9s} {len(sub):4d}t | میانگین p {sub['p'].mean():.2f} | "
                  f"برد واقعی {sub['win'].mean():.1%} | میانگین r {sub['r'].mean():+.3f}")
    # دلتای ماهانه نسبت به بدون وزن
    oos2 = oos.assign(mon=pd.to_datetime(oos["entry_time"]).dt.strftime("%Y-%m"))
    base_m = oos2.groupby("mon")["pnl"].sum()
    for mcol, name in (("m_stepped", "پله‌ای"), ("m_linear", "خطی"),
                       ("m_kelly", "کلی")):
        w = (oos2[mcol] * oos2["pnl"]).groupby(oos2["mon"]).sum()
        delta = (w - base_m).dropna()
        print(f"  دلتای ماهانه {name}: بهتر از بدون‌وزن در "
              f"{int((delta > 0).sum())}/{len(delta)} ماه | جمع "
              f"{delta.sum():+,.0f}$")
    per.assign(dataset=label).to_csv(
        save / f"phase6b_windows_{label.split()[0].lower()}.csv", index=False)
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۶-ب: سایزینگ احتمالاتی")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    cfg = MetaWFOConfig()
    print("═══ ۱) داده ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka_m1 = load_duka(find(pathlib.Path(args.duka).name))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka_m1))
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    regime = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)

    rows = []
    pipeline(m15, h4, regime, sess, "MT5", cfg, rows, save)

    print("\n═══ ۲) تکرار روی Dukascopy (منبع مستقل) ═══")
    d15 = to_m15(duka_m1)
    d4 = resample_tf(duka_m1, "4h")
    dreg = RegimeEngine().compute(d15)["regime"].to_numpy()
    dsess = np.asarray(session_open(d15["time"]), dtype=bool)
    duka_arms = pipeline(d15, d4, dreg, dsess, "Duka", cfg, rows, save)

    # ---------- حکم میلهٔ از پیش ثبت‌شده ----------
    print("\n═══ ۳) حکم میلهٔ ثبت‌شده ═══")
    def best_arm(arms, skip_oracle=True):
        cands = [a for a in arms if a["pf"] == a["pf"] and a["pf"] < 10
                 and "اوراکل" not in a["arm"] and "گیت" not in a["arm"]
                 and "take_all" not in a["arm"]]
        return max(cands, key=lambda a: a["pf"]) if cands else None

    mt5_arms = [a for a in rows if a["dataset"] == "MT5"]
    b = best_arm(mt5_arms)
    duka_b = best_arm(duka_arms)
    take_all_duka = next(a for a in duka_arms
                         if "take_all" in a["arm"])
    if b is None:
        print("  هیچ بازوی وزنی معتبری تولید نشد.")
        return
    ok_mt5 = b["pf"] >= 1.4 and b["per_month"] >= 13
    ok_duka = (duka_b is not None and duka_b["pf"] >= 1.3
               and duka_b["pf"] > take_all_duka["pf"])
    print(f"  MT5 بهترین بازو: {b['arm']} → PF {b['pf']:.2f} "
          f"({b['per_month']:.1f}/ماه) | میله: PF≥1.4 و ≥13/ماه → "
          f"{'✅' if ok_mt5 else '❌'}")
    if duka_b:
        print(f"  duka بهترین بازو: {duka_b['arm']} → PF {duka_b['pf']:.2f} "
              f"| take_all: {take_all_duka['pf']:.2f} | میله: ≥1.3 و >take_all → "
              f"{'✅' if ok_duka else '❌'}")
    verdict = "✅ میله عبور شد — فاز ۶-ب قابل‌ارائه است (walk-forward بعدی + تأیید انسان)"
    if not (ok_mt5 and ok_duka):
        verdict = "❌ میله عبور نشد — ۶-ب هم دفن می‌شود؛ G6 ساده می‌ماند."
    print(f"\n  حکم: {verdict}")

    pd.DataFrame(rows).to_csv(save / "phase6b_results.csv", index=False)
    print(f"\n💾 ذخیره شد: {save / 'phase6b_results.csv'}")


if __name__ == "__main__":
    main()
