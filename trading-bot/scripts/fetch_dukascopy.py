"""دانلود تاریخچه M1 از Dukascopy — اجرا روی ماشین خودت (اینترنت آزاد).

مثال (ویندوز، از ریشه پروژه):
    py scripts/fetch_dukascopy.py --years 3
    py scripts/fetch_dukascopy.py --start 2023-01-01 --end 2026-09-01

خروجی: data/xauusd_m1_duka.csv.gz (فشرده و سبک)
بعد از اتمام، فایل را در چت Arena ضمیمه کن تا فاز ۱ (بک‌تست) با دیتای
عمیق و تمیز انجام شود. تعطیلات/آخر هفته خودکار رد می‌شوند.
"""
import argparse
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.data.dukascopy import DukascopyDownloader


def main() -> None:
    ap = argparse.ArgumentParser(description="دانلود M1 از Dukascopy")
    ap.add_argument("--instrument", default="XAUUSD")
    ap.add_argument("--years", type=float, default=3.0, help="چند سال اخیر (اگر --start ندادی)")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (پیش‌فرض: امروز)")
    ap.add_argument("--out", default=None, help="مسیر خروجی csv.gz")
    ap.add_argument("--cache", default="data/cache/duka", help="کش روزانه")
    ap.add_argument("--workers", type=int, default=3, help="نخ‌های همزمان — کم نگه دار تا 503 نخوری")
    ap.add_argument("--retries", type=int, default=6, help="تلاش برای هر روز روی خطای موقت (503 و...)")
    ap.add_argument("--retry-passes", type=int, default=3, help="پاس‌های تلاش مجدد برای روزهای جا‌مانده")
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=int(args.years * 365.25)))

    out = pathlib.Path(args.out or f"data/{args.instrument.lower()}_m1_duka.csv.gz")
    print(f"⬇️  دانلود {args.instrument} از {start} تا {end} (کش: {args.cache})")

    dl = DukascopyDownloader(args.instrument, cache_dir=args.cache, max_workers=args.workers)
    df = dl.fetch_range(start, end)

    if df.empty:
        raise SystemExit("❌ هیچ داده‌ای دریافت نشد — اینترنت/سیمبل را چک کن")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")

    # گزارش سلامت — چشم‌تکی: قیمت طلا باید در بازه منطقی تاریخ مربوطه باشد
    print(f"\n✅ {len(df):,} کندل M1 ذخیره شد: {out}")
    print(f"بازه: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"قیمت: min={df['low'].min():.2f}  max={df['high'].max():.2f}")
    by_year = df.groupby(df["time"].dt.year)["close"].mean().round(1)
    print("میانگین قیمت سالانه (برای چک سلامت):")
    print(by_year.to_string())
    print("\n📦 این فایل را در چت Arena ضمیمه کن (Attachment) تا بک‌تست فاز ۱ انجام شود.")


if __name__ == "__main__":
    main()
