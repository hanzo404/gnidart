"""آزمایش ایده‌های قرضی از EA خارجی «AI-MTF» (MQL5) — سپتامبر ۲۰۲۶.

پس‌زمینه: کاربر فрагمنتی از یک اکسپرت تجاری (KNN چند-تایم‌فریم + فیلتر H1
EMA + فیلتر شاخص دلار DXY + سایزینگ ریسک-محور + ریسک‌فری خودکار + تریلینگ
نقطه‌ای) فرستاد و پرسید: چی از این برای پروژهٔ ما قابل‌استفاده است؟

این اسکریپت فقط ایده‌های «قابل‌جدی‌گرفتن» را با همان استانداردهای خودمان
می‌آزماید — روی بک‌تست واقع‌گرایانهٔ G6 (فقط-خرید + گیت رژیم + سشن +
نردبان بریکر ۳/۴/۵/۶، دادهٔ MT5 سه‌ساله، ریسک ۰.۵٪):

    ۱) ریسک‌فری خودکار (be_at_frac): وقتی close به کسری از مسیر TP برسد،
       استاپ به نقطهٔ ورود می‌آید. آزمایش ۰.۵ و ۰.۳۳.
    ۲) فیلتر دلار (ایدهٔ DXY در EA): خرید طلا فقط وقتی دلار «در حال
       تقویت نباشد». چون DXY در دسترس نیست، از نرخ رسمی روزانهٔ ECB
       (USD→EUR) به‌عنوان پروکسی معکوس دلار استفاده می‌کنیم — EUR حدود
       ۵۷٪ وزن DXY است. علیّت: فیکسِ روزِ قبلِِ ورود، قبل از باز شدن
       کندلِ ورود منتشر شده است.
    ۳) ترکیب بهترین‌ها + تکرار روی منبع مستقل Dukascopy.
    ۴) آمار زمینه‌ای: هم‌بستگی روزانهٔ طلا با دلار؛ عملکرد معاملات
       پایه روی روزهای دلار-صعودی vs دلار-نزولی (تشخیصِ اینکه فیلتر
       اصلاً لبه دارد یا نه).

خروجی: marketdata/ea_ideas_results.csv

یادآوری صادقانه: این‌ها آزمایش‌های درون-نمونه (in-sample) روی یک مسیر
تاریخی‌اند. هر تغییری در رباتِ زنده فقط بعد از walk-forward + تأیید
انسان اعمال می‌شود — قانون پروژه.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.risk import BreakerPolicy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import RegimeEngine, session_open
from bot.risk.circuit_breaker import CircuitBreaker


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def load_dollar_fix() -> pd.Series:
    """نرخ‌های روزانهٔ رسمی ECB: USD→EUR (فیکس ~۱۴:۰۰ CET).

    usdeur بالا = دلار قوی‌تر (مثل DXY بالا).
    """
    path = find("ecb_usdeur_daily.json")
    d = json.load(open(path))
    s = pd.Series({pd.Timestamp(k): float(v)
                   for k, v in d["rates"].items()}).sort_index()
    assert s.index.is_monotonic_increasing
    return s


def dollar_ok_mask(times: pd.Series, fix: pd.Series) -> np.ndarray:
    """ماسک «ورود خرید مجاز است» برای هر کندل (تاریخ UTC کندلِ ورود).

    قانون (قرینهٔ فیلتر DXY در EA): خرید طلا فقط اگر دلار در آخرین
    تغییرِ روزانه‌اش در حال تقویت نباشد:
        allow(d) = fix[d−1] ≤ fix[d−2]
    که d−1 و d−2 دو فیکسِ «قبل از تاریخ کندل» هستند — یعنی هر دو قبل از
    باز شدن کندلِ ورود منتشر شده‌اند (ECB فیکس را ~۱۴:۰۰ CET همان روز
    می‌دهد؛ ما فقط فیکس‌های کاملاً گذشته را می‌بینیم).
    """
    fix_dates = fix.index.values
    fix_vals = fix.to_numpy()
    day = pd.DatetimeIndex(pd.to_datetime(times)).normalize()
    uniq, inv = np.unique(day.values, return_inverse=True)
    # آخرین فیکسِ «قبل از» هر تاریخ یکتا (strictly before)
    idx = np.searchsorted(fix_dates, uniq, side="left") - 1
    ok = np.zeros(len(uniq), dtype=bool)
    valid = idx >= 1          # حداقل دو فیکس گذشته لازم است
    ii = idx[valid]
    ok[valid] = fix_vals[ii] <= fix_vals[ii - 1]   # دلار در حال تقویت نیست
    return ok[inv]


def run_g6(m15, h4, reg_arr, sess_arr, *, be=None, eq=30000.0):
    cfg = BacktestConfig(start_equity=eq, risk_pct=0.005, max_positions=1,
                         cooldown_bars=1, slippage_usd=0.05,
                         spread_gate_usd=0.40, be_at_frac=be)
    pol = BreakerPolicy(CircuitBreaker(derate_at=3, deep_derate_at=4,
                                       pause_at=5, halt_at=6))
    strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg_arr, sess_arr,
                          direction=+1)
    return Backtester(m15, strat, cfg, risk_policy=pol).run()


def fmt(label: str, m: dict, months: float) -> str:
    return (f"  {label:34s} {m['n_trades']:4d}t | {m['n_trades']/months:5.1f}/ماه | "
            f"برد {m['win_rate']:.1%} | PF {m['profit_factor']:.2f} | "
            f"${m['total_pnl']:8,.0f} | DD {m['max_dd_pct']:5.1%} | "
            f"{m['expectancy_r']:.3f}R")


def main() -> None:
    ap = argparse.ArgumentParser(description="آزمایش ایده‌های EA خارجی")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()
    save = find(pathlib.Path(args.mt5).name).parent

    print("═══ ۰) داده + فیلتر دلار ═══")
    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka_m1 = load_duka(find(pathlib.Path(args.duka).name))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka_m1))
    m15 = to_m15(mt5_utc)
    h4 = resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)

    fix = load_dollar_fix()
    dmask = dollar_ok_mask(m15["time"], fix)
    print(f"  کندل M15: {len(m15):,} | فیلتر دلار: {dmask.mean():.1%} کندل‌ها مجاز")
    months = (m15["time"].iloc[-1] - m15["time"].iloc[0]).days / 30.44

    # ---------- ۱) A/B ریسک‌فری خودکار ---------- #
    print("\n═══ ۱) ریسک‌فری خودکار (be_at_frac) — MT5، G6 ═══")
    rows = []

    def add(label, res, variant):
        m = res.metrics
        print(fmt(label, m, months))
        t = res.trades
        be_saved = int(((t["reason"] == "stop") & (t["r"] > -0.5)).sum()) if len(t) else 0
        rows.append({"variant": variant, "n": m["n_trades"],
                     "per_month": round(m["n_trades"] / months, 1),
                     "wr": round(m["win_rate"], 3),
                     "pf": round(m["profit_factor"], 2),
                     "pnl": round(m["total_pnl"]),
                     "dd": round(m["max_dd_pct"], 3),
                     "exp_r": round(m["expectancy_r"], 3),
                     "be_scratch": be_saved})
        return res

    base = add("A: پایه (بدون BE)", run_g6(m15, h4, reg, sess), "base")
    add("B: BE در ۰.۵ مسیر TP", run_g6(m15, h4, reg, sess, be=0.5), "be_0.50")
    add("C: BE در ۰.۳۳ مسیر TP", run_g6(m15, h4, reg, sess, be=0.33), "be_0.33")

    # ---------- ۲) فیلتر دلار ---------- #
    print("\n═══ ۲) فیلتر دلار (ECB USD→EUR روزانه) — MT5، G6 ═══")
    sess_dollar = sess & dmask
    add("D: فقط وقتی دلار در تقویت نیست",
        run_g6(m15, h4, reg, sess_dollar), "dollar")
    add("E: فیلتر دلار + BE ۰.۵",
        run_g6(m15, h4, reg, sess_dollar, be=0.5), "dollar_be_0.50")

    # تشخیص: معاملات پایه روی روزهای دلار-صعودی چه‌طور بودند؟
    bt = base.trades
    if len(bt):
        dmask_tr = dollar_ok_mask(bt["entry_time"], fix)
        for name, sel in (("دلار در تقویت (بلاک‌شده)", ~dmask_tr),
                          ("دلار خنثی/نزولی", dmask_tr)):
            sub = bt[sel]
            if len(sub):
                print(f"    {name}: {len(sub)}t | برد {(sub['pnl'] > 0).mean():.1%} | "
                      f"میانگین {sub['r'].mean():+.3f}R | ${sub['pnl'].sum():,.0f}")
            else:
                print(f"    {name}: 0t")

    # ---------- ۳) تکرار روی Dukascopy (منبع مستقل) ---------- #
    print("\n═══ ۳) تکرار روی Dukascopy ═══")
    d15 = to_m15(duka_m1)
    d4 = resample_tf(duka_m1, "4h")
    dreg = RegimeEngine().compute(d15)["regime"].to_numpy()
    dsess = np.asarray(session_open(d15["time"]), dtype=bool)
    ddollar = dollar_ok_mask(d15["time"], fix)
    dmonths = (d15["time"].iloc[-1] - d15["time"].iloc[0]).days / 30.44

    def add_duka(label, res, variant):
        m = res.metrics
        print(fmt(label, m, dmonths))
        t = res.trades
        be_saved = int(((t["reason"] == "stop") & (t["r"] > -0.5)).sum()) if len(t) else 0
        rows.append({"variant": variant, "n": m["n_trades"],
                     "per_month": round(m["n_trades"] / dmonths, 1),
                     "wr": round(m["win_rate"], 3),
                     "pf": round(m["profit_factor"], 2),
                     "pnl": round(m["total_pnl"]),
                     "dd": round(m["max_dd_pct"], 3),
                     "exp_r": round(m["expectancy_r"], 3),
                     "be_scratch": be_saved})

    add_duka("F: duka پایه", run_g6(d15, d4, dreg, dsess), "duka_base")
    add_duka("G: duka + BE ۰.۵", run_g6(d15, d4, dreg, dsess, be=0.5),
             "duka_be_0.50")
    add_duka("H: duka + فیلتر دلار", run_g6(d15, d4, dreg, dsess & ddollar),
             "duka_dollar")

    # ---------- ۴) آمار زمینه‌ای: طلا ↔ دلار ---------- #
    print("\n═══ ۴) هم‌بستگی روزانهٔ طلا با دلار (زمینهٔ کلی) ═══")
    gold_day = m15.groupby(m15["time"].dt.normalize())["close"].last()
    fx = fix.to_frame("usdeur")
    fx["gold"] = gold_day.reindex(fx.index).ffill()
    fx = fx.dropna()
    rets = fx.pct_change().dropna()
    corr = rets["usdeur"].corr(rets["gold"])
    print(f"  corr(تغییر دلار, تغییر طلا) = {corr:+.2f}  "
          f"({len(rets)} روز) — منفی یعنی دلار قوی ← طلا ضعیف")
    print(f"  سهم روزهای «دلار در تقویت» = {(rets['usdeur'] > 0).mean():.0%}")
    g_up = rets.loc[rets["usdeur"] > 0, "gold"].mean() * 100
    g_dn = rets.loc[rets["usdeur"] <= 0, "gold"].mean() * 100
    print(f"  میانگین تغییر طلا: روز دلار-قوی {g_up:+.2f}% | روز دلار-ضعیف {g_dn:+.2f}%")

    out = pd.DataFrame(rows)
    out.to_csv(save / "ea_ideas_results.csv", index=False)
    print(f"\n💾 ذخیره شد: {save / 'ea_ideas_results.csv'}")


if __name__ == "__main__":
    main()
