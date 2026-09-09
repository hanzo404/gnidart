"""MT5 Heartbeat — اولین اجرای واقعی: تست اتصال، اکانت دمو، جریان کندل زنده.

اجرا (ویندوز، از ریشه پروژه):
    py scripts/mt5_heartbeat.py
    py scripts/mt5_heartbeat.py --symbol EURUSD.r --count 300

پیش‌نیاز: ترمینال MT5 باز باشد و با اکانت دمو لاگین شده باشد.
اگر همه‌چیز سبز باشد، خروجی را برای منتور بفرست (کپی متن یا اسکرین‌شات).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.data.mt5_data import MT5DataProvider, _HAS_MT5


def main() -> None:
    if not _HAS_MT5:
        raise SystemExit("❌ پکیج MetaTrader5 نصب نیست: py -m pip install MetaTrader5")

    ap = argparse.ArgumentParser(description="تست اتصال ربات به متاتریدر ۵")
    ap.add_argument("--symbol", default="XAUUSD", help="نام سیمبل در بروکر تو")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--count", type=int, default=500)
    args = ap.parse_args()

    with MT5DataProvider(symbol=args.symbol) as prov:
        acc = prov.account_summary()
        print("=" * 60)
        print(f" اکانت  : {acc['login']} @ {acc['server']}")
        print(f" بالانس : {acc['balance']:,.2f} {acc['currency']} | لوریج 1:{acc['leverage']}")
        if acc["is_demo"]:
            print(" نوع    : ✅ دمو (درست همینه)")
        else:
            print(" نوع    : ⚠️⚠️ غیر دمو! فعلاً فقط دمو — پول واقعی قفل است")
        print("=" * 60)

        q = prov.live_quote()
        print(f"{args.symbol}: bid={q['bid']}  ask={q['ask']}  "
              f"spread={q['spread_points']:.1f} points")

        df = prov.candles(args.timeframe, args.count, closed_only=True)
        print(f"\n{len(df)} کندل بسته‌شده {args.timeframe} | "
              f"بازه: {df.iloc[0]['time']}  →  {df.iloc[-1]['time']}")
        print(df.tail(3).to_string(index=False))

        out = pathlib.Path("data") / f"mt5_{args.symbol.replace('.', '_')}_{args.timeframe}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n✅ ذخیره شد: {out}")
        print("HEARTBEAT OK — قلب ربات می‌تپد. 🫀 این خروجی را برای منتور بفرست.")


if __name__ == "__main__":
    main()
