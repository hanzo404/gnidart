"""فاز ۱ — اعتبارسنجی داده + شبیه‌سازی استراتژی v0 روی XAUUSD (۳ سال).

اجرا (روی هر دو ویندوزِ تو و اینجا):
    py scripts/run_phase1.py
    py scripts/run_phase1.py --mt5 data/xauusd_m1_mt5_3y.csv.gz --duka data/xauusd_m1_duka.csv.gz

چهار سناریو:
    A) v0 «همان‌طور که نوشته شده» روی M1: بدون گیت اسپرد، بدون چک پوزیشن
       (سقف ۲۰ پوزیشن همزمان — همان انفجار واقعی DB)، کول‌داون ۵ کندل
    B) v0 + نگهبان‌های حداقلی روی M1: حداکثر ۱ پوزیشن + گیت اسپرد $0.40
    C) همان B اما روی M15 (تایم‌فریم قفل‌شدهٔ پروژه)
    D) همان C روی دادهٔ Dukascopy (UTC) با پروفایل اسپرد ساعتی از MT5 —
       اعتبارسنجی متقاطع نتیجه بین دو منبع مستقل
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import (BacktestConfig, Backtester, print_metrics)
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (attach_spread_profile, infer_offset_minutes,
                              load_duka, load_mt5, resample_tf, shift_mt5_to_utc,
                              spread_profile_by_hour, to_m15)


def find(name: str) -> pathlib.Path:
    """اول data/ پروژه، بعد ../marketdata/ (سندباکس)."""
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد (data/ یا marketdata/ را چک کن)")


def validate_report(name: str, df: pd.DataFrame) -> None:
    flat = 0
    if "volume" in df.columns:
        f = ((df["volume"] == 0) & (df["open"] == df["close"])
             & (df["high"] == df["low"]))
        flat = int(f.sum())
    print(f"  {name:<22} {len(df):>10,} کندل | {df['time'].min()} → {df['time'].max()}"
          f" | NaN={int(df.isna().sum().sum())} | تخت={flat:,}")


def cross_validate(mt5_utc: pd.DataFrame, duka: pd.DataFrame) -> dict:
    """هم‌دقیقه‌های دو منبع: اختلاف قیمت باید چند سنت باشد نه چند دلار."""
    mt = mt5_utc["time"].to_numpy()
    mc = mt5_utc["close"].to_numpy()
    dt = duka["time"].to_numpy()
    dc = duka["close"].to_numpy()
    idx = np.searchsorted(dt, mt)
    idx_c = np.clip(idx, 0, len(dt) - 1)
    hit = dt[idx_c] == mt
    diff = np.abs(mc[hit] - dc[idx_c][hit])
    # فقط دقیقه‌های متوالی برای همبستگی بازده
    both = (np.diff(mt[hit]) == np.timedelta64(1, "m"))
    r_mt = np.diff(mc[hit])[both]
    r_dk = np.diff(dc[idx_c][hit])[both]
    corr = float(np.corrcoef(r_mt, r_dk)[0, 1]) if both.sum() > 100 else float("nan")
    return {"matched": int(hit.sum()), "mt5_only": int((~hit).sum()),
            "med_diff": float(np.median(diff)), "p95_diff": float(np.percentile(diff, 95)),
            "corr": corr}


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۱: اعتبارسنجی + شبیه‌سازی v0")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    ap.add_argument("--save-dir", default=None,
                    help="پوشهٔ ذخیرهٔ معاملات (پیش‌فرض: کنار فایل داده)")
    args = ap.parse_args()

    mt5_path = pathlib.Path(args.mt5) if pathlib.Path(args.mt5).exists() else find(pathlib.Path(args.mt5).name)
    duka_path = pathlib.Path(args.duka) if pathlib.Path(args.duka).exists() else find(pathlib.Path(args.duka).name)

    print("═══ ۱) بارگذاری و اعتبارسنجی دو منبع ═══")
    mt5 = load_mt5(mt5_path)
    duka = load_duka(duka_path)
    validate_report("MT5 (سرور)", mt5)
    validate_report("Dukascopy (UTC)", duka)

    print("\n═══ ۲) تشخیص اختلاف ساعت سرور MT5 با UTC ═══")
    offsets = infer_offset_minutes(mt5, duka)
    vals = sorted(set(offsets.values()))
    print(f"  آفست‌های ماهانه: {vals} دقیقه"
          + (f" → ثابت +{vals[0]} دقیقه" if len(vals) == 1 else
             " → سرور DST دارد (تابستان/زمستان فرق می‌کند)"))
    odd = {k: v for k, v in offsets.items() if v != pd.Series(list(offsets.values())).mode()[0]}
    if odd:
        print(f"  ماه‌های غیرعادی: {odd}")
    mt5_utc = shift_mt5_to_utc(mt5, offsets)

    print("\n═══ ۳) اعتبارسنجی متقاطع (MT5-UTC در برابر Dukascopy) ═══")
    cv = cross_validate(mt5_utc, duka)
    print(f"  دقیقه‌های مشترک: {cv['matched']:,} | فقط در MT5: {cv['mt5_only']:,}")
    print(f"  اختلاف close هم‌دقیقه: میانه ${cv['med_diff']:.3f} | ۹۵٪ زیر ${cv['p95_diff']:.3f}")
    print(f"  همبستگی بازده M1: {cv['corr']:.4f}")
    m15m, m15d = to_m15(mt5_utc), to_m15(duka)
    print(f"  کندل M15: MT5={len(m15m):,} | duka={len(m15d):,}")

    print("\n═══ ۴) پروفایل اسپرد (میانهٔ ساعتی از MT5، دلار) ═══")
    prof = spread_profile_by_hour(mt5_utc)
    best_h, worst_h = int(prof.idxmin()), int(prof.idxmax())
    print(f"  بهترین ساعت UTC {best_h:02d}:00 (${prof.min():.2f}) | "
          f"بدترین {worst_h:02d}:00 (${prof.max():.2f}) | میانه ${prof.median():.2f}")

    # ------------------- داده‌های هر سناریو ------------------- #
    h4_mt5 = resample_tf(mt5, "4h")        # سرور-تایم مثل خود v0
    h4_duka = resample_tf(duka, "4h")
    m15_mt5 = to_m15(mt5)
    m15_duka = attach_spread_profile(to_m15(duka), prof)

    print("\n═══ ۵) شبیه‌سازی‌ها ═══")
    runs = [
        ("A — v0 همان‌طور که نوشته شد (M1، بدون نگهبان)",
         mt5, BacktestConfig(fixed_lots=0.01, max_positions=20,
                             cooldown_bars=5, slippage_usd=0.05)),
        ("B — v0 + نگهبان‌ها (M1، ۱ پوزیشن، گیت اسپرد)",
         mt5, BacktestConfig(fixed_lots=0.01, max_positions=1,
                             cooldown_bars=5, slippage_usd=0.05,
                             spread_gate_usd=0.40)),
        ("C — v0 + نگهبان‌ها (M15، تایم‌فریم پروژه)",
         m15_mt5, BacktestConfig(fixed_lots=0.01, max_positions=1,
                                 cooldown_bars=1, slippage_usd=0.05,
                                 spread_gate_usd=0.40)),
        ("D — همان C روی منبع Dukascopy (اعتبارسنجی متقاطع)",
         m15_duka, BacktestConfig(fixed_lots=0.01, max_positions=1,
                                  cooldown_bars=1, slippage_usd=0.05,
                                  spread_gate_usd=0.40)),
    ]
    results = {}
    for title, bars, cfg in runs:
        strat = V0Strategy(h4_mt5 if bars is not m15_duka else h4_duka)
        res = Backtester(bars, strat, cfg).run()
        print_metrics(title, res)
        results[title[0]] = res

    # ------------------- جدول جمع‌بندی ------------------- #
    print("\n═══ ۶) جمع‌بندی ═══")
    print(f"{'سناریو':<6}{'معاملات':>10}{'برد%':>8}{'PF':>7}{'سود کل$':>12}{'افتاکس%':>10}")
    for k, res in results.items():
        m = res.metrics
        if m.get("n_trades", 0) == 0:
            print(f"{k:<6}{'0':>10}")
            continue
        print(f"{k:<6}{m['n_trades']:>10,}{m['win_rate']:>8.1%}"
              f"{m['profit_factor']:>7.2f}{m['total_pnl']:>12,.0f}"
              f"{m['max_dd_pct']:>10.1%}")

    # ذخیرهٔ معاملات برای تحلیل‌های بعدی
    save_dir = pathlib.Path(args.save_dir) if args.save_dir else mt5_path.parent
    try:
        for k, res in results.items():
            if not res.trades.empty:
                res.trades.to_csv(save_dir / f"phase1_{k}_trades.csv.gz",
                                  index=False, compression="gzip")
        print(f"\n💾 معاملات ذخیره شد در: {save_dir}")
    except OSError as e:
        print(f"\n(ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
