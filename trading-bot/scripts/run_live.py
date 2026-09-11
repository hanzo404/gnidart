"""فاز ۵ — اجرای دموی فوروارد روی MT5 (فقط ویندوز، فقط اکانت دمو).

اجرا (روی ویندوزِ تو، از ریشهٔ پروژه):
    py scripts/run_live.py --dry-run     # روز اول: همه‌چیز جز ارسال سفارش
    py scripts/run_live.py               # دموی واقعی (سفارش‌های واقعی روی دمو)
    py scripts/run_live.py --once        # فقط یک چرخه (تست)
    py scripts/run_live.py --ack         # رفع halt بعد از بازبینی

ترمینال MT5 باید باز و به اکانت دمو لاگین باشد (گام‌های SETUP).
Ctrl+C = توقف تمیز. استاپ/تارگت سروری‌اند؛ خاموشی ربات = بدون ریسک یتیمی.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.config import BotConfig
from bot.journal.store import Journal
from bot.live.runner import LiveRunner


def main() -> None:
    ap = argparse.ArgumentParser(description="دموی فوروارد — فاز ۵")
    ap.add_argument("--symbol", default=None, help="پیش‌فرض از config")
    ap.add_argument("--dry-run", action="store_true",
                    help="همه‌چیز جز ارسال سفارش (پوزیشن کاغذی)")
    ap.add_argument("--once", action="store_true", help="فقط یک چرخه")
    ap.add_argument("--ack", action="store_true",
                    help="رفع halt بریکر بعد از بازبینی انسانی")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--state", default="data/live_state.json")
    args = ap.parse_args()

    cfg = BotConfig.load(args.config)
    symbol = args.symbol or (cfg.trading.symbols[0] if cfg.trading.symbols
                             else "XAUUSD")

    # ---- وایرینگ (فقط این‌جا import های MT5 انجام می‌شوند) ----
    from bot.data.mt5_data import MT5DataProvider
    from bot.execution.mt5_adapter import MT5ExecutionAdapter

    provider = MT5DataProvider(symbol)
    provider.connect()
    try:
        acc = provider.account_summary()
        print(f"🏦 اکانت: {acc['login']} روی {acc['server']} | "
              f"موجودی ${acc['balance']:,.0f} | "
              f"{'✅ دمو' if acc['is_demo'] else '⛔ واقعی!'}")
        if not acc["is_demo"]:
            raise SystemExit("⛔ این اکانت دمو نیست — اجرا متوقف شد. "
                             "به اکانت دمو لاگین کن.")
        q = provider.live_quote()
        print(f"💱 {symbol}: bid {q['bid']:.2f} / ask {q['ask']:.2f} | "
              f"اسپرد {q['spread_points']:.0f} پوینت")

        adapter = MT5ExecutionAdapter(symbol, magic=cfg.live.magic)
        journal = Journal(cfg.journal.path)
        runner = LiveRunner(provider, adapter, journal, cfg,
                            state_path=args.state, symbol=symbol,
                            dry_run=args.dry_run)

        if args.ack:
            runner.acknowledge()
            print("✅ halt رفع شد (حجم کاهش‌یافته حفظ می‌شود)")
            return

        mode = "DRY-RUN (کاغذی)" if args.dry_run else "دموی واقعی"
        print(f"\n🤖 فاز ۵ | حالت: {mode} | سیمبل {symbol} | M15")
        print(f"   فیلتر جهت: {cfg.live.direction_filter} | "
              f"ریسک: {cfg.risk.risk_per_trade:.1%}/معامله × نردبان بریکر")
        print(f"   ژورنال: {cfg.journal.path} | وضعیت: {args.state}")
        print("   Ctrl+C = توقف تمیز (استاپ‌ها سروری‌اند)\n")

        poll = cfg.live.poll_seconds
        while True:
            try:
                msg = runner.on_cycle()
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{stamp}] {msg}")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — یک چرخهٔ خراب کل ربات را نکشد
                print(f"[!] خطای چرخه (ادامه می‌دهیم): {e}")
            if args.once:
                break
            time.sleep(poll)
    finally:
        provider.close()
        print("\n👋 اتصال بسته شد.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C = توقف عمدی؛ پیام خداحافظی را finallyِ داخل main چاپ کرده
        # (👋 اتصال بسته شد). این‌جا فقط جلوی traceback زائد پایتون را
        # می‌گیریم تا خروجی تمیز باشد.
        pass
