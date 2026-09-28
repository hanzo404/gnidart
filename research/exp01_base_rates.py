"""آزمایش ۰۱ — نرخ پایهٔ ادعاهای ICT.

پرسش: آیا «نقدشوندگی‌ربایی» (sweep) و «شکاف ارزش منصفانه» (FVG) در دادهٔ
واقعی، همان بازدهی‌ای را که روایت ICT می‌گوید تولید می‌کنند؟

این آزمایش «نرخ پایه» است: هنوز هیچ ورود/خروج/هزینه‌ای در کار نیست.
اگر اثر در این مرحله صفر باشد، هیچ‌چیزِ بعدیِ ساختاری نجاتش نمی‌دهد.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ictlab.data import SPECS, load_m1, resample
from ictlab.structure import StructureParams, atr, detect_sweeps, detect_fvg
from ictlab.stats import bootstrap_tstat

pd.set_option("display.width", 220)
HORIZONS = [1, 2, 3, 5, 10, 20]


def fwd_r(df: pd.DataFrame, sig_bar: np.ndarray, direction: np.ndarray, atr_: np.ndarray,
          horizons=HORIZONS) -> dict[int, np.ndarray]:
    """بازده آینده بر حسب R (ATR-نرمال) برای هر افق."""
    close = df.close.values
    n = len(df)
    out = {}
    for h in horizons:
        fut = np.full(n, np.nan)
        ok = sig_bar + h < n
        idx = sig_bar[ok]
        fut[idx] = (close[idx + h] - close[idx]) * direction[ok] / atr_[idx]
        out[h] = fut
    return out


def report(name: str, r_by_h: dict[int, np.ndarray], label: str) -> pd.DataFrame:
    rows = []
    for h, r in r_by_h.items():
        r = r[~np.isnan(r)]
        if len(r) < 30:
            continue
        se = r.std() / np.sqrt(len(r))
        rows.append({"setup": label, "symbol": name, "horizon": h, "n": len(r),
                     "mean_R": r.mean(), "t": r.mean() / se if se else 0,
                     "win%": 100 * (r > 0).mean(), "med_R": np.median(r)})
    return pd.DataFrame(rows)


def main() -> None:
    p = StructureParams()
    frames = []
    for name, sp in SPECS.items():
        m1 = load_m1(sp)
        m5 = resample(m1, 5)
        a5 = atr(m5, p.m1_atr)
        print(f"\n{'='*100}\n{name}: M1={len(m1):,}  M5={len(m5):,}")

        # ── شاهد: بازده تصادفی هم‌تعداد (پایهٔ مقایسه)
        rng = np.random.default_rng(11)
        rnd_bars = rng.choice(len(m5), size=min(20000, len(m5)), replace=False)
        rnd_bars.sort()
        rnd = fwd_r(m5, rnd_bars, np.ones(len(rnd_bars), dtype=int), a5)
        frames.append(report(name, rnd, "RANDOM-1bar"))

        # ── ادعای ۱: نقدشوندگی‌ربایی
        sw = detect_sweeps(m5, p, a5)
        if sw.empty:
            print("  no sweeps"); continue
        bars = sw.bar.values
        dirs = sw.dir.values.astype(int)
        s = fwd_r(m5, bars, dirs, a5)
        frames.append(report(name, s, "SWEEP-reversal"))
        # و جهت مخالف (چه می‌شود اگر خلافش معامله کنیم؟)
        s2 = fwd_r(m5, bars, -dirs, a5)
        frames.append(report(name, s2, "SWEEP-with-sweep"))

        # ── ادعای ۲: FVG
        fv = detect_fvg(m5, p, a5)
        if not fv.empty:
            f = fwd_r(m5, fv.bar.values, fv.dir.values.astype(int), a5)
            frames.append(report(name, f, "FVG-continuation"))
            # بازگشت به شکاف: آیا قیمت واقعاً ناحیهٔ خالی را پر می‌کند؟
            top, bot, d = fv.top.values, fv.bot.values, fv.dir.values.astype(int)
            filled = np.zeros(len(fv), dtype=bool)
            hi_a, lo_a = m5.high.values, m5.low.values
            for i, b in enumerate(fv.bar.values):
                end = min(b + 60, len(m5))
                seg_h, seg_l = hi_a[b + 1:end], lo_a[b + 1:end]
                if len(seg_h) == 0:
                    continue
                filled[i] = (seg_l.min() <= bot[i]) if d[i] == 1 else (seg_h.max() >= top[i])
            bars_to_fill = []
            for i, b in enumerate(fv.bar.values):
                end = min(b + 60, len(m5))
                for j in range(b + 1, end):
                    if d[i] == 1 and lo_a[j] <= bot[i]:
                        bars_to_fill.append(j - b); break
                    if d[i] == -1 and hi_a[j] >= top[i]:
                        bars_to_fill.append(j - b); break
            print(f"  FVG: {len(fv):,}  |  ۶۰ کندل بعد پر شد: {100*filled.mean():.1f}%  |  "
                  f"میانهٔ زمان تا پر شدن: {np.median(bars_to_fill) if bars_to_fill else float('nan'):.0f} کندل")
            # آیا پر شدنِ FVG خودش پیش‌بین است؟ (وقتی قیمت به FVG برگشت، کجا می‌رود؟)
            touch_dir, touch_idx = [], []
            for i, b in enumerate(fv.bar.values):
                end = min(b + 60, len(m5))
                for j in range(b + 1, end):
                    if d[i] == 1 and lo_a[j] <= bot[i]:
                        touch_idx.append(j); touch_dir.append(d[i]); break
                    if d[i] == -1 and hi_a[j] >= top[i]:
                        touch_idx.append(j); touch_dir.append(d[i]); break
            if touch_idx:
                t = fwd_r(m5, np.array(touch_idx), np.array(touch_dir), a5)
                frames.append(report(name, t, "FVG-on-touch-continuation"))

        # ── ادعای ۳: اثر ساعت روز (killzone)
        sw2 = sw.copy()
        sw2["hour"] = pd.to_datetime(sw2.time).dt.hour
        byh = []
        for h_, g in sw2.groupby("hour"):
            idx = g.bar.values
            fr = fwd_r(m5, idx, g.dir.values.astype(int), a5)
            r = fr[5]
            r = r[~np.isnan(r)]
            if len(r) < 100:
                continue
            byh.append({"hour_utc": h_, "n": len(r), "mean_R@5": r.mean(), "win%@5": 100*(r>0).mean(),
                        "t": r.mean()/(r.std()/np.sqrt(len(r)))})
        if byh:
            bh = pd.DataFrame(byh).sort_values("mean_R@5", ascending=False)
            print("\n  ── اثر ساعت UTC روی بازده ۵ کندلی بعد از جاروب ──")
            print(bh.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
            bh.to_csv(f"results/hour_effect_{name}.csv", index=False)

    out = pd.concat(frames, ignore_index=True)
    out["t_boot"] = [bootstrap_tstat(r) for r in
                     [np.array([]) for _ in out.index]] if False else 0.0
    out.to_csv("results/exp01_base_rates.csv", index=False)
    print(f"\n{'='*100}\nنتیجهٔ کلی (افق ۵ کندلی M5):")
    key = out[out.horizon == 5].copy()
    key = key[["symbol", "setup", "n", "mean_R", "t", "win%", "med_R"]].sort_values(["symbol", "mean_R"], ascending=[True, False])
    print(key.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
