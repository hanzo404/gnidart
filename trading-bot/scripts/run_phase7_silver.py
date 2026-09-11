"""فاز ۷ — چند-سمبلی: XAGUSD (نقره) — نگاه اول با طرح از-پیش-ثبت‌شده.

اجرا:
    py scripts/run_phase7_silver.py            (بعد از رسیدن دادهٔ نقره)

زمینه: «معاملهٔ بیشتر» از مسیر شل‌کردن گیت‌ها سه بار مرده (P5b، ۶-الف،
۶-ب). مسیر درستِ فرکانس = تنوع نماد. نقره خواهرِ طلاست: همان درایور
دلاری، همان سشن‌های نقدینگی، نوسانِ (و اسپردِ) بزرگ‌ترِ نسبی.

طرحِ این نگاه اول — قبل از دیدن هر نتیجه‌ای ثبت شده (۲۰۲۶-۰۹-۱۱):
    - استراتژی v0 و همهٔ گیت‌ها «عیناً» مثل طلا؛ هیچ تنظیم مجددی نه.
    - فقط هزینه‌ها مقیاس می‌شوند با ضریب k = میانه ATR14 نقره ÷ میانه
      ATR14 طلا (هر دو M15، همان پنجرهٔ سه‌ساله):
        sl_pad = 0.50×k | slippage = 0.05×k
    - اسپرد نقره نامعلوم است (داده duka فقط bid دارد) → فرض ثابت
      $0.04 پایه و $0.08 استرس — هر دو گزارش می‌شوند.
    - جهت: خرید و فروش جداگانه (تشخیص؛ تصمیمِ جهت مثل طلا فقط با شواهد).
    - بدون بریکر (نگاه اول، سنجهٔ لبه است نه نردبان)؛ لات ثابت 0.01؛
      contract = 5000 اونس (استاندارد نقره).
    - میلهٔ «ادامه‌دادن به WFO کامل»: PF ≥ 1.15 در نمونهٔ کامل با
      ≥ ۳ معامله/ماه و مثبت‌بودن در ≥ ۲ سال از ۳ سال. کمتر از این →
      نقره کنار گذاشته می‌شود و گزارشش صادقانه ثبت می‌گردد.
    - نکتهٔ مقایسه: بک‌تست‌های duka-طلا ستون اسپرد نداشتند (اسپرد=0)؛
      این‌جا برای نقره اسپرد فرضی تزریق می‌کنیم — محافظه‌کارانه‌تر از
      رفرنس طلا؛ مقایسهٔ عددی با احتیاط.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import load_duka, resample_tf, to_m15
from bot.regime.engine import RegimeEngine, session_open


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(
        f"❌ فایل {name} پیدا نشد.\n"
        "   روی ماشین خودت اجرا کن:\n"
        "     py scripts/fetch_dukascopy.py --instrument XAGUSD "
        "--start 2023-09-11 --end 2026-09-09\n"
        "   بعد خروجی (data/xagusd_m1_duka.csv.gz) را در چت ضمیمه کن.")


def atr14_median(m15: pd.DataFrame) -> float:
    h, l, c = m15["high"], m15["low"], m15["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return float(atr.median())


def run(m15, h4, reg, sess, direction, sl_pad, slip, spread_usd,
        label, months):
    m = m15.copy()
    m["spread_usd"] = spread_usd
    strat = GatedStrategy(V0Strategy(h4, rr=2.5, sl_pad=sl_pad),
                          reg, sess, direction=direction)
    res = Backtester(m, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, contract_oz=5000.0,
        max_positions=1, cooldown_bars=1, slippage_usd=slip,
        spread_gate_usd=None)).run()
    mm = res.metrics
    print(f"  {label:34s} {mm['n_trades']:4d}t | {mm['n_trades']/months:5.1f}/ماه | "
          f"برد {mm['win_rate']:.1%} | PF {mm['profit_factor']:.2f} | "
          f"${mm['total_pnl']:8,.0f} | DD {mm['max_dd_pct']:5.1%} | "
          f"{mm['expectancy_r']:.3f}R")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="فاز ۷: نقره — نگاه اول")
    ap.add_argument("--silver", default="data/xagusd_m1_duka.csv.gz")
    ap.add_argument("--gold", default="data/xauusd_m1_duka.csv.gz")
    ap.add_argument("--spread-base", type=float, default=0.04)
    ap.add_argument("--spread-stress", type=float, default=0.08)
    args = ap.parse_args()
    save = find(pathlib.Path(args.gold).name).parent

    print("═══ ۱) داده ═══")
    sil = load_duka(find(pathlib.Path(args.silver).name))
    gold = load_duka(find(pathlib.Path(args.gold).name))
    s15, s4 = to_m15(sil), resample_tf(sil, "4h")
    g15 = to_m15(gold)
    months = (s15["time"].iloc[-1] - s15["time"].iloc[0]).days / 30.44
    print(f"  نقره: {len(sil):,} کندل M1 → {len(s15):,} کندل M15 | "
          f"{s15['time'].iloc[0].date()} → {s15['time'].iloc[-1].date()} "
          f"({months:.0f} ماه)")
    print(f"  قیمت نقره: {sil['low'].min():.2f} → {sil['high'].max():.2f}")
    print("  میانگین سالانه:")
    print(sil.groupby(sil["time"].dt.year)["close"].mean().round(2).to_string())

    # ---------- ۲) مقیاس هزینه‌ها (از-پیش-ثبت‌شده: نسبت ATR) ----------
    k = atr14_median(s15) / atr14_median(g15)
    sl_pad, slip = 0.50 * k, 0.05 * k
    print(f"\n═══ ۲) مقیاس هزینه‌ها ═══")
    print(f"  میانه ATR14: نقره {atr14_median(s15):.3f} | طلا "
          f"{atr14_median(g15):.2f} → k = {k:.3f}")
    print(f"  sl_pad = 0.50×k = ${sl_pad:.3f} | slippage = 0.05×k = ${slip:.3f}")
    print(f"  اسپرد فرضی: پایه ${args.spread_base} | استرس ${args.spread_stress}")

    # ---------- ۳) رژیم و گیت‌ها (بدون هیچ تنظیم مجدد) ----------
    reg = RegimeEngine().compute(s15)["regime"].to_numpy()
    sess = np.asarray(session_open(s15["time"]), dtype=bool)
    from bot.regime.engine import NAMES
    dist = pd.Series(reg).map(NAMES).value_counts(normalize=True)
    print(f"\n═══ ۳) رژیم (همان آستانه‌های طلا) ═══")
    print("  " + " | ".join(f"{i}: {v:.0%}" for i, v in dist.items()))

    # ---------- ۴) نگاه اول ----------
    for spread, tag in ((args.spread_base, "اسپرد پایه"),
                        (args.spread_stress, "اسپرد استرس")):
        print(f"\n═══ ۴) نگاه اول — {tag} ═══")
        run(s15, s4, reg, sess, +1, sl_pad, slip, spread,
            f"فقط خرید + گیت کامل ({tag})", months)
        run(s15, s4, reg, sess, -1, sl_pad, slip, spread,
            f"فقط فروش + گیت کامل ({tag})", months)

    # سال‌به‌سالِ خریدِ گیت‌شده (تمرکز سود را ببینیم)
    print("\n═══ ۵) خریدِ گیت‌شده — سال‌به‌سال (اسپرد پایه) ═══")
    res = run(s15, s4, reg, sess, +1, sl_pad, slip, args.spread_base,
              "خرید گیت‌شده (مرجع بخش ۵)", months)
    t = res.trades.assign(year=pd.to_datetime(res.trades["entry_time"]).dt.year)
    print(t.groupby("year")["pnl"].agg(["count", "sum"]).round(0).to_string())

    # ---------- ۶) حکم میله ----------
    mm = res.metrics
    years_pos = int((t.groupby("year")["pnl"].sum() > 0).sum())
    ok = (mm["profit_factor"] >= 1.15 and mm["n_trades"] / months >= 3
          and years_pos >= 2)
    print(f"\n═══ ۶) حکم میلهٔ نگاه اول ═══")
    print(f"  PF {mm['profit_factor']:.2f} (≥1.15؟) | "
          f"{mm['n_trades']/months:.1f}/ماه (≥3؟) | "
          f"سال‌های مثبت {years_pos}/۳ (≥2؟) → "
          f"{'✅ ادامه: WFO کامل' if ok else '❌ نقره کنار — گزارش ثبت شد'}")

    try:
        pd.DataFrame([{"pf": mm["profit_factor"],
                       "n": mm["n_trades"],
                       "per_month": round(mm["n_trades"] / months, 1),
                       "wr": round(mm["win_rate"], 3),
                       "pnl": round(mm["total_pnl"]),
                       "k_atr": round(k, 4),
                       "sl_pad": round(sl_pad, 4),
                       "years_pos": years_pos}]).to_csv(
            save / "phase7_silver_firstlook.csv", index=False)
        print(f"\n💾 {save / 'phase7_silver_firstlook.csv'}")
    except OSError as e:
        print(f"  (ذخیره نشد: {e})")


if __name__ == "__main__":
    main()
