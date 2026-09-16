"""دکترِ آستین‌های لایو — تشخیص فقط-خواندنی؛ به بات دست نمی‌زند.

اجرا (فقط روی VPS؛ امن در حالی که رانرها روشن‌اند):
    py scripts/live_doctor.py            # عکس فوری از همهٔ گیت‌ها (یک‌باره)
    py scripts/live_doctor.py --watch    # ناظر سشن: هر کندل تازه را مثل خود
                                        # رانر داوری می‌کند + اسپرد هر دقیقه
                                        # را ثبت می‌کند؛ تا ۲۰:۰۰ UTC خودش
                                        # می‌ایستد و خلاصه می‌دهد (Ctrl+C هم)

زمینه (۱۶ سپتامبر ۲۰۲۶): بک‌تستِ همان کانفیگ در بازهٔ لایو ۵ معامله
می‌گرفت (analysis_live_gap) و بازپخشِ مسیر تصمیمِ لایو روی دیتای تاریخی
همهٔ گیت‌ها را سبز نشان داد (analysis_live_replay) — ولی لایو ۰ معامله
دارد. دکترِ یک‌باره state/پوزیشن/مشخصات/تاریخچه را تبرئه کرد؛ تنها
شکاف باقی‌مانده: اسپرد «لحظه‌ای» در لحظهٔ تصمیم داخل سشن (کندل‌ها
مینیممِ اسپرد را ثبت می‌کنند، رانر لحظه‌ای را چک می‌کند). ناظر همین
شکاف را پر می‌کند: اگر لحظهٔ «آمادهٔ ورود» بیاید و بات واقعی وارد
نشود، تفاوت از سمت رانر است؛ اگر ناظر هم بلاک شود، محیط است.

تنها فایلی که می‌نویسد: data/doctor_watch.csv (خروجی خودش — نه ژورنال،
نه state، نه سفارش).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.v0_strategy import V0Strategy
from bot.config import BotConfig
from bot.live.runner import eet_dst_active, validate_symbol_spec
from bot.regime.engine import (NAMES as REGIME_NAMES, TREND_UP, RegimeEngine,
                               session_open)

SLEEVES = ("XAUUSD", "XAGUSD")
WATCH_CSV = pathlib.Path("data/doctor_watch.csv")


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
    base = s.get("sleeve_base")
    bits.append(f"sleeve_base=${base:,.0f}" if base is not None else "sleeve_base=—")
    print(" | ".join(bits) if bits else "خالی")


def decide(m15: pd.DataFrame, h4: pd.DataFrame, profile, prv) -> str:
    """همان زنجیرهٔ _maybe_enter — خروجی: حکم یا اولین گیتِ بلاک‌کننده."""
    t = pd.Timestamp(m15["time"].iloc[-1])
    now_server = t + pd.Timedelta(minutes=15)
    off = 180 if eet_dst_active(now_server.to_pydatetime()) else 120
    entry_utc = now_server - pd.Timedelta(minutes=off)
    if not bool(session_open(pd.Series([entry_utc]))[0]):
        return f"خارج از سشن ({entry_utc:%H:%M} UTC)"
    try:
        q = prv.live_quote()
    except Exception as e:  # noqa: BLE001
        return f"بلاک: تیک ناموجود ({e})"
    sp = float(q["spread_points"]) * float(q["point"])
    tick = q.get("tick_epoch")
    if tick and time.time() - float(tick) > 120:
        return "بلاک: دادهٔ کهنه (آخرین تیک > ۱۲۰ ثانیه)"
    if sp > profile.max_spread_usd:
        return f"بلاک: اسپرد ${sp:.2f} > گیت ${profile.max_spread_usd}"
    r_now = int(RegimeEngine().compute(m15)["regime"].iloc[-1])
    inner = V0Strategy(h4, rr=2.5, sl_pad=profile.sl_pad)
    inner.prepare(m15)
    sig = inner.on_bar(len(m15) - 1)
    if sig is None:
        return f"بدون سیگنال (رژیم {REGIME_NAMES.get(r_now)})"
    if sig.direction <= 0:
        return f"سیگنال {sig.direction:+d} خلاف فیلتر فقط-خرید"
    if r_now != TREND_UP:
        return f"بلاک: رژیم {REGIME_NAMES.get(r_now)} ≠ TREND_UP"
    return (f"✅ آمادهٔ ورود! (اسپرد ${sp:.2f} | استاپ "
            f"{sig.stop:.{profile.digits}f}) — اگر بات وارد نشد، مشکل سمت رانر است")


def summarize(rows: list) -> None:
    if not rows:
        print("(داده‌ای ثبت نشد)")
        return
    df = pd.DataFrame(rows)
    try:
        WATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(WATCH_CSV, index=False, encoding="utf-8")
        print(f"💾 جزئیات کامل: {WATCH_CSV}")
    except OSError as e:
        print(f"(CSV نوشته نشد: {e})")
    for sym in SLEEVES:
        d = df[df["symbol"] == sym]
        if d.empty:
            continue
        print(f"\n── خلاصهٔ {sym} ──")
        bars = d[d["kind"] == "bar"]
        if not bars.empty:
            vc = bars["verdict"].value_counts()
            print(f"  کندل‌های داوری‌شده: {len(bars)}")
            for v, n in vc.items():
                print(f"    {n:3d}× {v}")
            ready = bars[bars["verdict"].str.startswith("✅")]
            for _, r in ready.iterrows():
                print(f"  ⚡ لحظهٔ آماده: {r['ts_utc']} → {r['verdict']}")
        ticks = d[d["kind"] == "tick"]
        if not ticks.empty:
            sp = ticks["spread_usd"]
            gate = float(d["gate"].dropna().iloc[0]) if d["gate"].notna().any() else float("nan")
            print(f"  تیک‌های اسپرد داخل سشن: {len(ticks)} | میانه "
                  f"${sp.median():.2f} | p95 ${sp.quantile(.95):.2f} | "
                  f"بیشینه ${sp.max():.2f} | بالای گیت: "
                  f"{(sp > gate).mean():.0%} (گیت ${gate:.2f})")


def run_watch(cfg: BotConfig) -> None:
    from bot.data.mt5_data import MT5DataProvider
    import MetaTrader5 as mt5

    sleeves = {}
    for sym in SLEEVES:
        prv = MT5DataProvider(sym)
        prv.connect()
        sleeves[sym] = (prv, cfg.profile_for(sym))
    magic = cfg.live.magic

    print("═══ ناظر سشن — مثل خود رانر داوری می‌کند؛ فقط-خواندنی برای بات ═══")
    print("هر ۶۰ ثانیه یک اسکن. اسپرد فقط داخل سشن (۱۲–۲۰ UTC) نمونه‌برداری"
          " می‌شود. پایان خودکار بعد از ۲۰:۰۰ UTC؛ Ctrl+C = خلاصهٔ فوری.\n")
    ti = mt5.terminal_info()
    if ti is not None and not getattr(ti, "trade_allowed", False):
        print("⛔ توجه: Algo Trading ترمینال خاموش است — لحظه‌های «آمادهٔ "
              "ورود» سفارشان رد می‌شود! اول دکمهٔ Algo Trading در MT5 را "
              "سبز کن.\n")

    rows: list = []
    last_bar: dict = {}
    saw_session = False
    try:
        while True:
            now = datetime.now(timezone.utc)
            in_sess = 12 <= now.hour < 20
            if in_sess:
                saw_session = True
            elif saw_session and now.hour >= 20:
                print("\n── سشن تمام شد (۲۰:۰۰ UTC) ──")
                break
            for sym, (prv, p) in sleeves.items():
                if in_sess:
                    try:
                        q = prv.live_quote()
                        sp = float(q["spread_points"]) * float(q["point"])
                        rows.append({"ts_utc": f"{now:%Y-%m-%d %H:%M:%S}",
                                     "symbol": sym, "kind": "tick",
                                     "spread_usd": sp,
                                     "gate": p.max_spread_usd, "verdict": ""})
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    m15 = prv.candles("M15", cfg.live.history_bars,
                                      closed_only=True)
                    h4 = prv.candles("H4", 150, closed_only=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{now:%H:%M:%S}] {sym}: کندل ناموجود ({e})")
                    continue
                nb = str(m15["time"].iloc[-1]) if len(m15) else None
                if not nb or nb == last_bar.get(sym):
                    continue
                last_bar[sym] = nb
                ours = [x for x in (mt5.positions_get(symbol=sym) or [])
                        if x.magic == magic]
                if ours:
                    v = "بلاک: پوزیشن باز در ترمینال (سقف ۱)"
                elif len(m15) < 250:
                    v = "بلاک: تاریخچهٔ کم (<۲۵۰ کندل)"
                else:
                    v = decide(m15, h4, p, prv)
                    if v.startswith("✅"):
                        ti = mt5.terminal_info()
                        if ti is not None and not getattr(ti, "trade_allowed",
                                                          False):
                            v += " — ⚠️ ولی Algo Trading خاموش: رد می‌شود!"
                rows.append({"ts_utc": f"{now:%Y-%m-%d %H:%M:%S}",
                             "symbol": sym, "kind": "bar",
                             "spread_usd": None, "gate": p.max_spread_usd,
                             "verdict": v})
                print(f"[{now:%H:%M:%S}] {sym} بار {nb} → {v}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n── توقف دستی ──")
    summarize(rows)


def run_once(cfg: BotConfig) -> None:
    print("═══ دکتر آستین‌های لایو — فقط-خواندنی ═══")
    now = datetime.now(timezone.utc)
    print(f"الان {now:%Y-%m-%d %H:%M} UTC | سشن ورود: ۱۲–۲۰ UTC "
          f"({'داخل سشن' if 12 <= now.hour < 20 else 'خارج سشن — پیام «خارج از سشن» در این ساعت طبیعی است'})")

    print("\n── ۱) فایل‌های state (data/) ──")
    show_state(pathlib.Path("data/live_state.json"))          # طلا
    show_state(pathlib.Path("data/live_state_xagusd.json"))   # نقره

    from bot.data.mt5_data import MT5DataProvider
    import MetaTrader5 as mt5

    print("\n── ۲) اکانت ──")
    prov = MT5DataProvider(SLEEVES[0])
    prov.connect()
    acc = prov.account_summary()
    print(f"اکانت {acc['login']} روی سرور {acc['server']} | "
          f"{'✅ دمو' if acc['is_demo'] else '⛔ واقعی!'} | "
          f"موجودی ${acc['balance']:,.0f}")

    # درس ۱۶ سپتامبر: ۲۶ روزِ صفرِ معامله = دکمهٔ Algo Trading خاموش
    # (retcode 10027 — ترمینال سفارش برنامه‌ای را رد می‌کند). از این پس
    # این چک همیشه اولین خط گزارش است.
    ti = mt5.terminal_info()
    ta = bool(getattr(ti, "trade_allowed", False)) if ti else None
    print("Algo Trading ترمینال: "
          + ("✅ روشن — سفارش‌ها قابل ارسال" if ta else
             "⛔ خاموش — همهٔ سفارش‌ها با retcode 10027 رد می‌شوند! "
             "(دکمهٔ Algo Trading در نوار ابزار MT5 را سبز کن)"))

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

        # اسپرد لحظه‌ای
        q = prv.live_quote()
        sp_usd = float(q["spread_points"]) * float(q["point"])
        ok = sp_usd <= p.max_spread_usd
        print(f"اسپرد لحظه‌ای: ${sp_usd:.3f} "
              + ("✓ زیر گیت" if ok else f"⛔ بالای گیت ${p.max_spread_usd}"))

        # مشخصات نماد
        warns = validate_symbol_spec(p, prv.symbol_spec())
        print("مشخصات نماد: " + ("✅ سازگار با پروفایل" if not warns
                                 else "⚠️ " + "؛ ".join(warns)))

        # تاریخچه + اسپردِ کندل‌های سشن اخیر
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

        # رژیم و سیگنال روی پنجرهٔ جاری (اطلاعی)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="تشخیص فقط-خواندنی آستین‌های لایو")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--watch", action="store_true",
                    help="ناظر سشن: داوری هر کندل تازه مثل رانر + اسپرد هر دقیقه")
    args = ap.parse_args()
    cfg = BotConfig.load(args.config)
    if args.watch:
        run_watch(cfg)
    else:
        run_once(cfg)


if __name__ == "__main__":
    main()
