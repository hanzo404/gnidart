"""خروجی تاریخچه M1 سیمبل از ترمینال MT5 خودت — اجرا روی ویندوز.

قبل از اجرا (برای گرفتن حداکثر تاریخچه):
    MT5 → Tools → Options → Charts → Max bars in chart = Unlimited
    سپس ترمینال را یک‌بار بسته و باز کن.

مثال:
    py scripts/export_mt5_history.py                      # XAUUSD، حداکثر موجود
    py scripts/export_mt5_history.py --symbol XAUUSD.r

خروجی: data/xauusd_m1_mt5.csv.gz — همان فیدِ بروکر خودت؛
هم برای بک‌تست (دید واقعی بروکر) و هم برای اعتبارسنجی متقاطع با Dukascopy.
بعد از اتمام، فایل را در چت Arena ضمیمه کن.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.data.mt5_data import MT5DataProvider, _HAS_MT5


def main() -> None:
    if not _HAS_MT5:
        raise SystemExit("❌ پکیج MetaTrader5 نصب نیست: py -m pip install MetaTrader5")

    ap = argparse.ArgumentParser(description="خروجی تاریخچه M1 از MT5")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--count", type=int, default=2_000_000, help="حداکثر تعداد کندل درخواستی")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with MT5DataProvider(symbol=args.symbol) as prov:
        df = prov.candles("M1", args.count, closed_only=True)

    if df.empty:
        raise SystemExit("❌ داده‌ای برنگشت — Max bars in chart را Unlimited کن و ترمینال را ری‌استارت کن")

    out = pathlib.Path(args.out or f"data/{args.symbol.lower().replace('.', '_')}_m1_mt5.csv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")

    print(f"✅ {len(df):,} کندل M1 بسته‌شده ذخیره شد: {out}")
    print(f"بازه: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"میانگین حجم تیک: {df['tick_volume'].mean():.0f}")
    print("\n📦 این فایل را در چت Arena ضمیمه کن (Attachment).")


if __name__ == "__main__":
    main()
