"""دکترِ آستین‌های لایو — تشخیص فقط-خواندنی؛ هیچ چیزی نمی‌نویسد/سفارش نمی‌دهد.

اجرا (فقط روی VPS؛ امن در حالی که رانرها روشن‌اند):
    py scripts/live_doctor.py

زمینه (۱۶ سپتامبر ۲۰۲۶): بک‌تستِ همان کانفیگ در بازهٔ لایو ۵ معامله
می‌گرفت (analysis_live_gap) و بازپخشِ مسیر تصمیمِ لایو روی همان دیتا
همهٔ گیت‌ها را سبز نشان داد (analysis_live_replay) — ولی لایو ۰ معامله
دارد. پس یکی از گیت‌های «محیطی» بسته است. این اسکریپت تک‌تک آن‌ها را
جداگانه چک می‌کند:

  ۱) state هر آستین (halt / پوزیشن کاغذی گیرکرده / بریکر)
  ۲) اکانت و بروکر + دمو بودن
  ۳) پوزیشن‌های باز ترمینال — مال ربات (magic) گیتِ «سقف ۱» را می‌بندد
  ۴) اسپرد لحظه‌ای + اسپردِ کندل‌های سشنِ اخیر در برابر گیت هر نماد
  ۵) تطبیق مشخصات نماد با پروفایل (digits/contract/volume)
  ۶) حجم تاریخچهٔ M15/H4 برای گرم‌شدن
  ۷) رژیم و سیگنال روی پنجرهٔ جاری (اطلاعی)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.v0_strategy import V0Strategy
from bot.config import BotConfig
from bot.live.runner import eet_dst_active, validate_symbol_spec
from bot.regime.engine import (NAMES as REGIME_NAMES, TREND_UP, RegimeEngine)

SLEEVES = ("XAUUSD", "XAGUSD")


def show_state(path: pathlib.Path) -> None:
    print(f"  {path.name}: ", end="")
    if not path.exists():
        print("وجود ندارد")
        return
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ خواندن نشد: {e}")
        return
    bits = []
    if s.get("paper") is not None:
        bits.append(f"⚠️ پوزیشن کاغذی باز! {s['paper']}")
    b = s.get("breaker") or {}
    if b.get("halted"):
        bits.append("⛔ HALT فعال")
    if b.get("paused_until"):
        bits.append(f"توقف تا {b.get('paused_until')}")
    if b.get("streak"):
        bits.append(f"streak={b.get('streak')}")
    bits.append(f"last_bar={s.get('last_bar_time')}")
    bits.append(f"sleeve_base=${s.get('sleeve_base', 0):,.0f}"
                if s.get("sleeve_base") is not None else "sleeve_base=—")
    print(" | ".join(bits) if bits else "خالی")


def main() -> None:
    ap = argparse.ArgumentParser(description="تشخیص فقط-خواندنی آستین‌های لایو")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = BotConfig.load(args.config)

    print("═══ دکتر آستین‌های لایو — فقط-خواندنی ═══")
    now = datetime.now(timezone.utc)
    print(f"الان {now:%Y-%m-%d %H:%M} UTC | سشن ورود: ۱۲–۲۰ UTC "
          f"({'داخل سشن' if 12 <= now.hour < 20 else 'خارج سشن — پیام «خارج از سشن» در این ساعت طبیعی است'})")

    print("\n── ۱) فایل‌های state (data/) ──")
    show_state(pathlib.Path("data/live_state.json"))          # طلا
    show_state(pathlib.Path("data/live_state_xagusd.json"))   # نقره

    try:
        from bot.data.mt5_data import MT5DataProvider
        import MetaTrader5 as mt5
    except ImportError as e:
        sys.exit(f"\n❌ MetaTrader5 فقط روی ویندوز است — این را روی VPS اجرا کن ({e})")

    print("\n── ۲) اکانت ──")
    prov = MT5DataProvider(SLEEVES[0])
    prov.connect()
    acc = prov.account_summary()
    print(f"اکانت {acc['login']} روی سرور {acc['server']} | "
          f"{'✅ دمو' if acc['is_demo'] else '⛔ واقعی!'} | "
          f"موجودی ${acc['balance']:,.0f}")

    print("\n── ۳) پوزیشن‌های باز ترمینال ──")
    allp = mt5.positions_get() or []
    ours = [p for p in allp if p.magic == cfg.live.magic]
    verdict = ("⛔ گیت «سقف ۱» بسته است — ربات معاملهٔ خودش را باز می‌بیند!"
               if ours else "✅ ربات پوزیشن باز ندارد (سقف ۱ آزاد)")
    print(f"کل اکانت: {len(allp)} | مال ربات (magic {cfg.live.magic}): "
          f"{len(ours)} — {verdict}")
    for p in allp:
        tag = " ← مال ربات" if p.magic == cfg.live.magic else ""
        print(f"  {p.symbol} {p.volume:g} لات @ {p.price_open} | "
              f"pnl {p.profit:+.2f}$ | magic={p.magic}{tag}")

    for sym in SLEEVES:
        p = cfg.profile_for(sym)
        print(f"\n── {sym} ──")
        prv = MT5DataProvider(sym)
        prv.connect()

        # ۴) اسپرد لحظه‌ای
        q = prv.live_quote()
        sp_usd = float(q["spread_points"]) * float(q["point"])
        ok = sp_usd <= p.max_spread_usd
        print(f"اسپرد لحظه‌ای: ${sp_usd:.3f} "
              + ("✓ زیر گیت" if ok else f"⛔ بالای گیت ${p.max_spread_usd}"))

        # ۵) مشخصات نماد
        warns = validate_symbol_spec(p, prv.symbol_spec())
        print("مشخصات نماد: " + ("✅ سازگار با پروفایل" if not warns
                                 else "⚠️ " + "؛ ".join(warns)))

        # ۶) تاریخچه + اسپردِ کندل‌های سشن اخیر
        m15 = prv.candles("M15", cfg.live.history_bars, closed_only=True)
        h4 = prv.candles("H4", 150, closed_only=True)
        warm = len(m15) >= 250
        print(f"تاریخچه: M15={len(m15)} کندل "
              + ("✓" if warm else "⛔ کمتر از ۲۵۰ — گرم نمی‌شود!")
              + f" | H4={len(h4)} کندل")
        if len(m15):
            c = m15.copy()
            # کندل‌ها به ساعتِ سرورند؛ برای فیلتر سشن (UTC) تبدیل کن
            off = 180 if eet_dst_active(datetime.now()) else 120
            c["utc_hour"] = (pd.to_datetime(c["time"])
                             - pd.Timedelta(minutes=off)).dt.hour
            c["sp_usd"] = c["spread"] * float(q["point"])
            sess = c[(c["utc_hour"] >= 12) & (c["utc_hour"] < 20)]
            if len(sess):
                med = sess["sp_usd"].median()
                p95 = sess["sp_usd"].quantile(0.95)
                flag = ("✓" if p95 <= p.max_spread_usd else
                        "⚠️ بیش از نصف سشن بالای گیت است"
                        if med > p.max_spread_usd else "~ مرزی")
                print(f"اسپرد کندل‌های سشن اخیر ({len(sess)} کندل): "
                      f"میانه ${med:.3f} | p95 ${p95:.3f} | "
                      f"بیشینه ${sess['sp_usd'].max():.3f} "
                      f"(گیت ${p.max_spread_usd}) {flag}")

        # ۷) رژیم و سیگنال روی پنجرهٔ جاری (اطلاعی)
        if warm and len(h4) >= 10:
            reg = RegimeEngine().compute(m15)
            r_now = int(reg["regime"].iloc[-1])
            inner = V0Strategy(h4, rr=2.5, sl_pad=p.sl_pad)
            inner.prepare(m15)
            sig = inner.on_bar(len(m15) - 1)
            sd = f"{sig.direction:+d}" if sig else "—"
            need = REGIME_NAMES[TREND_UP]
            have = REGIME_NAMES.get(r_now, str(r_now))
            print(f"وضعیت کندل آخر: رژیم {have} | سیگنال {sd} "
                  f"(برای خرید هر دو باید باشند: سیگنال +1 و رژیم {need})")


if __name__ == "__main__":
    main()
