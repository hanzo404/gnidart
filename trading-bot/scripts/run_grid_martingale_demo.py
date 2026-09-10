"""نمایشی از «۴۰٪ ماهانه» — گرید/مارتینگل روی دیتای واقعی طلا.

سؤال: روشی که وین‌ریت بالا + سود ماهانه‌ی بزرگ «می‌سازد» چیه و سرانجامش چیست؟

جواب کوتاه: گرید/مارتینگل — همان معماری EA-هایی که با «۹۵٪ وین‌ریت!»
فروش می‌شوند. این اسکریپت یکی از آن‌ها را وفادارانه می‌سازد:

    - خرید با لات پایه؛ هر $step پایین‌تر → لات × mult (بدون استاپ!)
    - سبد وقتی می‌بندد که قیمت به «سر‌به‌سر وزنی + $tp» برسد
    - مرگ = استاپ‌اوت مارجین (equity ≤ 50٪ مارجین مصرفی)

اجرای آن روی XAUUSD M15 سه‌ساله با چند تاریخ شروع مختلف نشان می‌دهد:
بازاری که صعودی است ماه‌های درخشان می‌سازد (همان بک‌تستی که فروشنده
نشانت می‌دهد) — تا روزی که یک حرکت واقعی، کل حساب را می‌بلعد.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.data.prepare import infer_offset_minutes, load_duka, load_mt5, shift_mt5_to_utc, to_m15


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


OZ = 100.0          # هر لات = 100 اونس


def run_grid(bars: pd.DataFrame, equity0: float, base: float, step: float,
             mult: float, tp: float, leverage: float = 100.0,
             stop_out: float = 0.5, spread: float = 0.15):
    """یک اجرای گرید-مارتینگل. خروجی: (زنده؟, ماه‌های بازده, تاریخ مرگ, DD)."""
    t = bars["time"].to_numpy()
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    bal = equity0
    lots = np.array([])          # لات‌های باز
    entries = np.array([])       # قیمت ورود هر سطح
    peak_eq = equity0
    max_dd = 0.0
    month_marks = []             # (ماه, balance_at_month_end)
    dead_at = None

    def floating(px: float) -> float:
        return float(((px - entries) * lots * OZ).sum()) if len(lots) else 0.0

    def margin_used(px: float) -> float:
        return float((lots * OZ * px / leverage).sum()) if len(lots) else 0.0

    for i in range(len(bars)):
        px_open = o[i] + spread
        # ۱) ورود اول / اضافه‌کردن سطوح (اگر low به سطح بعدی رسیده)
        if len(lots) == 0:
            lots = np.array([base]); entries = np.array([px_open])
        else:
            next_level = entries[-1] - step
            while l[i] <= next_level:            # ممکن است چند سطح در یک کندل
                lots = np.append(lots, lots[-1] * mult)
                entries = np.append(entries, next_level)
                # چک مرگ «حین اضافه‌کردن» — مثل لیکوییداسیون واقعی بروکر
                eq_lvl = bal + floating(next_level)
                if eq_lvl <= stop_out * margin_used(next_level):
                    dead_at = pd.Timestamp(t[i])
                    bal = max(eq_lvl, 0.0)   # نابودی کامل (بیش از این نیست)
                    break
                next_level -= step
            if dead_at is not None:
                break
        # ۲) چک مرگ (بدترین نقطه‌ی کندل — بدون سطح جدید هم ممکن است)
        eq_low = bal + floating(l[i])
        if len(lots) and eq_low <= stop_out * margin_used(l[i]):
            dead_at = pd.Timestamp(t[i])
            bal = max(eq_low, 0.0)           # بروکر همه را می‌بندد
            break
        # ۳) TP سبد: سر‌به‌سر وزنی + tp
        if len(lots):
            wavg = float((entries * lots).sum() / lots.sum())
            if h[i] >= wavg + tp:
                bal += (wavg + tp - entries) @ lots * OZ
                lots = np.array([]); entries = np.array([])
        # ۴) ثبت
        eq = bal + floating(float(o[i]))
        peak_eq = max(peak_eq, eq)
        max_dd = min(max_dd, eq / peak_eq - 1.0)
        month_marks.append((str(t[i])[:7], bal))

    # بازده ماهانه: ماه اول نسبت به سرمایه‌ی اولیه، بعدی‌ها زنجیره‌ای
    if len(month_marks):
        s = pd.Series(dict(month_marks)).groupby(level=0).last()
        vals = [equity0] + list(s.astype(float).values)
        rets = pd.Series(vals).pct_change().dropna()
        rets.index = list(s.index)[:len(rets)]
    else:
        rets = pd.Series(dtype=float)
    return dead_at is None, rets, dead_at, max_dd, bal


def main() -> None:
    ap = argparse.ArgumentParser(description="دموی گرید/مارتینگل")
    ap.add_argument("--mt5", default="data/xauusd_m1_mt5_3y.csv.gz")
    ap.add_argument("--duka", default="data/xauusd_m1_duka.csv.gz")
    args = ap.parse_args()

    mt5 = load_mt5(find(pathlib.Path(args.mt5).name))
    duka = load_duka(find(pathlib.Path(args.duka).name))
    m15 = to_m15(shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka)))
    print(f"داده: {m15['time'].iloc[0]} → {m15['time'].iloc[-1]} "
          f"({len(m15):,} کندل M15)\n")

    # سه پیکربندی «محبوب فروشگاه‌ها» + تاریخ شروع‌های مختلف
    configs = {
        "A: $10k, step $8, ×1.3": dict(equity0=10_000, base=0.01, step=8.0, mult=1.3, tp=3.0),
        "B: $3k,  step $5, ×1.5": dict(equity0=3_000, base=0.01, step=5.0, mult=1.5, tp=2.5),
        "C: $10k, step $5, ×1.2": dict(equity0=10_000, base=0.01, step=5.0, mult=1.2, tp=2.0),
    }
    starts = ["2023-09-11", "2024-01-02", "2024-07-01",
              "2025-01-02", "2025-07-01", "2026-01-02"]

    rows = []
    for cname, cfg in configs.items():
        for s in starts:
            b = m15[m15["time"] >= pd.Timestamp(s)].reset_index(drop=True)
            if len(b) < 500:
                continue
            alive, rets, dead_at, dd, bal = run_grid(b, **cfg)
            rows.append({
                "config": cname, "start": s,
                "alive": alive,
                "died": str(dead_at.date()) if dead_at is not None else "-",
                "months": len(rets),
                "best_month": rets.max() if len(rets) else float("nan"),
                "avg_month": rets.mean() if len(rets) else float("nan"),
                "final": round(bal),
            })
    res = pd.DataFrame(rows)
    print("═══ جدول بقا — ۶ تاریخ شروع × ۳ پیکربندی ═══")
    for cname, g in res.groupby("config"):
        print(f"\n── {cname}")
        print(g.drop(columns=["config"]).to_string(index=False,
              float_format=lambda v: f"{v:+.1%}" if abs(v) < 10 else f"{v:,.0f}"))
    save = find(pathlib.Path(args.mt5).name).parent
    res.to_csv(save / "grid_martingale_survival.csv", index=False)
    print(f"\n💾 marketdata/grid_martingale_survival.csv")

    # نمای «فروشنده»: بهترین ماه‌های بهترین پیکربندی
    best = res[res["alive"] == False].nlargest(1, "months")
    if len(best):
        r = best.iloc[0]
        b = m15[m15["time"] >= pd.Timestamp(r["start"])].reset_index(drop=True)
        for cname, cfg in configs.items():
            if cname == r["config"]:
                _, rets, _, _, _ = run_grid(b, **cfg)
                print(f"\n═══ نمای فروشنده: {cname} از {r['start']} ═══")
                print(f"ماه‌های زنده: {len(rets)} | بهترین ماه: "
                      f"{rets.max():+.1%} | میانگین: {rets.mean():+.1%} | "
                      f"مرگ: {r['died']}")


if __name__ == "__main__":
    main()
