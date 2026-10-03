"""مقایسهٔ خودکار پاریتی: لاگ MT5 در برابر مرجع پایتون.

روش کار:
  ۱) در MT5 یک تست کوتاه بگیرید و فایل لاگ ربات را بگیرید
     (مسیر معمول: MQL5/Files/  یا  Documents\\MT5\\...\\MQL5\\Files\\).
     یا از تب Journal چند سطر را در فایل متنی بچسبانید.
  ۲) این اسکریپت را اجرا کنید:
        python3 compare_parity.py --mt5 path/to/log.txt
  ۳) گزارش می‌گوید کدام سطرها یکی‌اند و کدام‌ها نه.

معیار: **جهت** باید یکی باشد. اختلاف زمانیِ چنددقیقه‌ای طبیعی است
(اختلاف «کندل جاری» بین Dukascopy و سرور بروکر).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# الگوی لاگ: زمان و «BUY»/«SELL» و قیمت
BUY_RE = re.compile(r"\bBUY\b", re.I)
SELL_RE = re.compile(r"\bSELL\b", re.I)
TIME_RE = re.compile(r"(\d{4})[.\-/](\d{2})[.\-/](\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?")
# قیمت‌ها: هم «ورود 34885.83» و هم «34885.83000»
NUM_RE = re.compile(r"\d+\.\d+|\d+")
# ترتیب فیلدهای عددی در لاگ: ورود، استاپ، هدف، R، حجم
ENTRY_RE = re.compile(r"ورود\s+([\d.]+)", re.I)
STOP_RE = re.compile(r"استاپ\s+([\d.]+)", re.I)


def parse_log(path: str) -> pd.DataFrame:
    """استخراج سطرهای BUY/SELL از لاگ."""
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        # سطرهای سیگنال: «شبیه‌سازی» یا «سفارش ثبت شد» یا «جاروب»
        if not re.search(r"شبیه\u200cسازی|سفارش ثبت شد|📝|✅", line):
            continue
        mt = TIME_RE.search(line)
        if not mt:
            continue
        g = mt.groups()
        ts = datetime(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]),
                       int(g[5]) if g[5] else 0)

        # ۱) اگر جهت صریح نوشته شده باشد (نسخهٔ جدید ربات)
        if BUY_RE.search(line) or SELL_RE.search(line):
            side = "BUY" if BUY_RE.search(line) else "SELL"
            em = ENTRY_RE.search(line)
            price = float(em.group(1)) if em else np.nan
        else:
            # ۲) نسخهٔ قدیمی: جهت از رابطهٔ ورود/استاپ استنباط می‌شود
            em, sm = ENTRY_RE.search(line), STOP_RE.search(line)
            if not (em and sm):
                continue
            e, st = float(em.group(1)), float(sm.group(1))
            if abs(e - st) < 1e-9:
                continue
            side = "BUY" if e > st else "SELL"
            price = e
        if not np.isfinite(price):
            continue
        rows.append({"time": ts, "dir": side, "price": price, "raw": line.strip()})
    return pd.DataFrame(rows)


def match(ref: pd.DataFrame, got: pd.DataFrame, tol_min: float, tol_price: float,
          max_rows: int = 40) -> None:
    print("\n" + "═" * 96)
    print("  مقایسهٔ پاریتی  (پایتون  ↔  MT5)")
    print("═" * 96)
    print(f"  مرجع پایتون: {len(ref)} معامله   |   لاگ MT5: {len(got)} سیگنال")
    print(f"  بازهٔ مرجع: {ref['time'].min()} → {ref['time'].max()}")
    print(f"  تحمل زمان: ±{tol_min:.0f} دقیقه   |   تحمل قیمت: ±{tol_price:.0f} واحد")
    print("─" * 96)

    # تطبیق سریع: مرتب‌سازی بر حسب زمان + جست‌وجوی دوجهته
    rt = ref["time"].values.astype("datetime64[ns]").astype("int64")
    gt = got["time"].values.astype("datetime64[ns]").astype("int64")
    order = np.argsort(gt)
    gs, gv = gt[order], got.iloc[order]
    gtd = gv["time"].values.astype("datetime64[ns]").astype("int64")
    tol_ns = int(tol_min) * 60 * 10**9

    used = np.zeros(len(gv), dtype=bool)
    results = []
    for i, r in ref.iterrows():
        lo, hi = np.searchsorted(gtd, rt[i] - tol_ns), np.searchsorted(gtd, rt[i] + tol_ns)
        best = -1
        for j in range(lo, hi):
            if used[j]:
                continue
            best = j
            break                      # نزدیک‌ترینِ نخستین در بازه (چون مرتب است)
        if best < 0:
            results.append((i, r, None, np.nan, "missing"))
            continue
        used[best] = True
        g = gv.iloc[best]
        dt = abs((g["time"].to_datetime64() - r["time"].to_datetime64())
                 .astype("timedelta64[s]").astype(float)) / 60.0
        same = g["dir"] == r["side"]
        # با فیلد entry_signal مقایسه می‌کنیم چون همان قیمتِ لیمیت است
        dprice = abs(float(g["price"]) - float(r["entry_signal"]))
        price_ok = dprice <= tol_price
        if same and price_ok:
            v = "ok"
        elif not same:
            v = "side"
        else:
            v = "price"
        results.append((i, r, g, dt, v))

    counts = {"ok": 0, "side": 0, "price": 0, "missing": 0}
    for *_, v in results:
        counts[v] += 1

    # چاپ: اولین چند سطر سالم و همهٔ سطرهای خراب
    bad_rows = [x for x in results if x[4] != "ok"]
    ok_rows = [x for x in results if x[4] == "ok"]
    show = ok_rows[:5] + bad_rows[:max_rows]
    print(f"  {'#':>4}  {'زمان مرجع':17} {'مرجع':5} {'MT5':5} {'جهت':5} "
          f"{'فاصله':>8} {'Δقیمت':>8}  نتیجه")
    print("─" * 96)
    for i, r, g, dt, v in show:
        if g is None:
            print(f"  {i + 1:>4}  {r['time'].strftime('%Y-%m-%d %H:%M'):17} {r['side']:5} "
                  f"{'—':5} {'—':5} {'—':>8} {'—':>8}  ⛔ سیگنال پیدا نشد")
            continue
        mark = {"ok": "✅ یکی", "side": "⛔ جهت متفاوت", "price": "⚠️ قیمت متفاوت"}[v]
        print(f"  {i + 1:>4}  {r['time'].strftime('%Y-%m-%d %H:%M'):17} {r['side']:5} "
              f"{g['dir']:5} {'✓' if g['dir'] == r['side'] else '✗':5} "
              f"{dt:7.1f}m {abs(float(g['price']) - float(r['entry_signal'])):8.2f}  {mark}")
    if len(bad_rows) > max_rows:
        print(f"  … و {len(bad_rows) - max_rows} سطر خرابِ دیگر")

    extra = int((~used).sum())
    total = len(ref)
    print("─" * 96)
    print(f"  ✅ یکی: {counts['ok']}/{total}   ⛔ جهت غلط: {counts['side']}   "
          f"⚠️ قیمت غلط: {counts['price']}   ⛔ پیدا نشد: {counts['missing']}   "
          f"سیگنالِ اضافه: {extra}")
    print("═" * 96)
    if counts["missing"] > 0 and extra == 0:
        print("  بازهٔ لاگ و مرجع یکی نیست — تاریخ شروع/پایان را چک کنید.")
    elif total:
        pct = 100 * counts["ok"] / total
        if pct >= 90:
            print(f"  🟢 پاریتی برقرار است ({pct:.0f}٪) → بروید مرحلهٔ بعد.")
        elif pct >= 60:
            print(f"  🟡 پاریتی مشکوک ({pct:.0f}٪) → لاگ هر دو را بفرستید، ادامه ندهید.")
        else:
            print(f"  🔴 پاریتی برقرار نیست ({pct:.0f}٪) → تست را متوقف کنید.")
    print("═" * 96)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5", required=True, help="مسیر فایل لاگ MT5")
    ap.add_argument("--ref", default="results/parity/US30_trades.csv",
                    help="فایل مرجعِ پایتون (ترجیحاً بازهٔ کوتاه)")
    ap.add_argument("--tol-min", type=float, default=5.0)
    ap.add_argument("--tol-price", type=float, default=10.0)
    a = ap.parse_args()

    if not os.path.exists(a.mt5):
        raise SystemExit(f"فایل لاگ پیدا نشد: {a.mt5}")
    ref = pd.read_csv(a.ref)
    ref["time"] = pd.to_datetime(ref["time"])
    # مرجع از پایتون می‌آید: dir عددی 1/-1 است. هر دو طرف به یک قالب
    # می‌روند وگرنه مقایسه همیشه «ناسازگار» می‌شود.
    ref["side"] = ref["dir"].map(lambda v: "BUY" if int(float(v)) == 1 else "SELL")
    for col in ("entry", "entry_signal"):
        if col not in ref.columns:
            raise SystemExit(
                f"ستون {col} در فایل مرجع نیست. فایل مرجع را دوباره بسازید:\n"
                "  python3 parity_check.py --symbol US30 --start 2023-09-04 --end 2023-09-15")
    got = parse_log(a.mt5)
    if got.empty:
        raise SystemExit(
            "هیچ سطر سیگنالی در لاگ پیدا نشد. اینها را چک کنید:\n"
            "  ۱) آیا Verbose=true بود؟ (پیش‌فرض در پریست هست)\n"
            "  ۲) آیا فایل CSV ربات را خوانده‌اید، نه فقط تب Journal؟\n"
            "  ۳) آیا نسخهٔ ربات شامل کلمهٔ BUY/SELL است؟ (نسخهٔ جدید)")
    match(ref, got, a.tol_min, a.tol_price)


if __name__ == "__main__":
    main()
