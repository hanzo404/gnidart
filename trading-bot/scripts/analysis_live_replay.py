"""بازپخش مسیر تصمیمِ لایو روی دیتای تاریخی — کدام گیت، ورودها را می‌خورَد؟

اجرا:
    py scripts/analysis_live_replay.py

زمینه (۱۶ سپتامبر ۲۰۲۶): بک‌تستِ همان کانفیگ در بازهٔ لایو (۲۱ آگوست تا
۱۰ سپتامبر) ۵ معامله می‌گرفت؛ لایو ۰. پس یکی از گیت‌های لایو با بک‌تست
رفتار متفاوت دارد. دو مظنون ساختاری:

  م۱) رژیم: لایو RegimeEngine را روی «پنجرهٔ ۸۰۰ کندلی» هر چرخه تازه
      می‌سازد؛ بک‌تست روی کل ۳ سال. ATR200 (تشخیص CHAOS) حافظهٔ بلند دارد.
  م۲) هم‌ترازی H4: بک‌تست کندل H4 را هم‌تراز UTC می‌سازد؛ بروکر MT5
      هم‌تراز ساعتِ سرور (EET/EEST) — گروه‌بندی متفاوتِ کندل‌ها =
      برچسب بایاس متفاوت (BULLISH/BEARISH).

آزمون‌ها:
  A) در «لحظهٔ تصمیم» هر ۵ معاملهٔ بک‌تست، دقیقاً مسیر گیت‌های لایو را
     می‌رویم و اولین گیتِ بلاک‌کننده را گزارش می‌کنیم.
  B) جاروب کل بازه: چند لحظهٔ «آمادهٔ ورود» در ۴ ترکیب می‌ماند؟
     B1 = h4 سرور + رژیم پنجره‌ای (نزدیک‌ترین به لایوِ واقعی)
     B2 = h4 UTC + رژیم پنجره‌ای        → اثر رژیم جدا می‌شود (B2−B3)
     B3 = h4 UTC + رژیم کامل            → دیدِ بک‌تست (صحت‌سنجی: ~۵)
     B4 = h4 سرور + رژیم کامل            → اثر H4 جدا می‌شود (B4−B3)

نکتهٔ هم‌ترازی: بازهٔ ما تماماً EEST (+3) است → مرزهای H4 سرور =
21:00, 01:00, 05:00… UTC (نیمه‌شب سرور). تغییر ساعت اروپا اواخر اکتبر
است، بیرون از بازه.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.backtest.engine import BacktestConfig, Backtester
from bot.backtest.gate import GatedStrategy
from bot.backtest.v0_strategy import V0Strategy
from bot.data.prepare import (infer_offset_minutes, load_duka, load_mt5,
                              resample_tf, shift_mt5_to_utc, to_m15)
from bot.regime.engine import (NAMES as REGIME_NAMES, TREND_UP,
                               RegimeEngine, session_open)

W0 = pd.Timestamp("2026-08-21")
W1 = pd.Timestamp("2026-09-10")
HIST = 800          # cfg.live.history_bars
H4HIST = 150        # provider.candles("H4", 150)
GATE = 0.40         # symbol_profiles XAUUSD.max_spread_usd


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def h4_server_aligned(m15: pd.DataFrame) -> pd.DataFrame:
    """کندل H4 با مرزهای ساعت سرور (نیمه‌شب EEST = 21:00 UTC تابستان)."""
    out = (m15.set_index("time")
           .resample("4h", label="left", closed="left",
                     origin=pd.Timestamp("2023-09-01 21:00:00"))
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last"}))
    return out.dropna().reset_index()


def signal_at(bars_win: pd.DataFrame, h4_frame: pd.DataFrame):
    s = V0Strategy(h4_frame.reset_index(drop=True), rr=2.5, sl_pad=0.50)
    s.prepare(bars_win)
    return s.on_bar(len(bars_win) - 1)


def main() -> None:
    mt5 = load_mt5(find("xauusd_m1_mt5_3y.csv.gz"))
    duka = load_duka(find("xauusd_m1_duka.csv.gz"))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka))
    m15 = to_m15(mt5_utc)
    h4_utc = resample_tf(mt5_utc, "4h")
    h4_srv = h4_server_aligned(m15)
    reg_full = RegimeEngine().compute(m15)["regime"].to_numpy()
    times = m15["time"].to_numpy()

    # بک‌تست مرجع (همان analysis_live_gap)
    strat = GatedStrategy(V0Strategy(h4_utc, rr=2.5),
                          reg_full, np.asarray(session_open(m15["time"]),
                                               dtype=bool), direction=+1)
    res = Backtester(m15, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, max_positions=1,
        cooldown_bars=1, slippage_usd=0.05, spread_gate_usd=GATE)).run()
    g = res.trades
    w = g[(g["entry_time"] >= W0) & (g["entry_time"] < W1)] \
        .sort_values("entry_time")
    print(f"═══ A) بازپخش {len(w)} معاملهٔ بک‌تست در مسیر تصمیمِ لایو ═══")

    def rname(v: int) -> str:
        return REGIME_NAMES.get(int(v), str(v))

    for k, (_, tr) in enumerate(w.iterrows(), 1):
        entry = pd.Timestamp(tr["entry_time"])
        dec_t = entry - pd.Timedelta(minutes=15)
        i = int(np.searchsorted(times, np.datetime64(dec_t)))
        if i >= len(m15) or pd.Timestamp(times[i]) != dec_t:
            print(f"\n#{k} ورود {entry} — کندل تصمیم پیدا نشد!")
            continue
        bars_win = m15.iloc[max(0, i - HIST + 1): i + 1]
        t = pd.Timestamp(times[i])
        h4u = h4_utc[h4_utc["time"] <= t].tail(H4HIST)
        h4s = h4_srv[h4_srv["time"] <= t].tail(H4HIST)
        reg_win = int(RegimeEngine().compute(bars_win)["regime"].iloc[-1])
        reg_fl = int(reg_full[i])
        sess_ok = bool(session_open(pd.Series([entry]))[0])
        sp = float(m15["spread_usd"].iloc[min(i + 1, len(m15) - 1)])
        sig_u = signal_at(bars_win, h4u)
        sig_s = signal_at(bars_win, h4s)
        sig_f = signal_at(bars_win, h4_utc[h4_utc["time"] <= t])
        print(f"\n#{k} ورود {entry} (تصمیم {dec_t:%H:%M}, pnl ${tr['pnl']:+.0f})")
        print(f"  سشن ورود {entry:%H:%M} UTC: {'✓' if sess_ok else '✗ خارج از ۱۲–۲۰'}"
              f" | اسپرد ${sp:.2f}: {'✓' if sp <= GATE else '✗ > گیت'}")
        for lbl, sig in (("h4-UTC (بک‌تست)", sig_u),
                         ("h4-سرور (بروکر)", sig_s),
                         ("h4-UTC-کامل", sig_f)):
            d = f"{sig.direction:+d}" if sig else "—"
            print(f"  سیگنال {lbl:16s}: {d}")
        print(f"  رژیم: پنجره‌ای={rname(reg_win)} | کامل={rname(reg_fl)}")

    # ── B) جاروب کل بازه ─────────────────────────────────────────
    print(f"\n═══ B) جاروب {W0.date()} → {W1.date()}: لحظات آمادهٔ ورود ═══")
    idx = np.where((m15["time"].to_numpy() >= np.datetime64(W0))
                   & (m15["time"].to_numpy() < np.datetime64(W1)))[0]
    counts = {"B1": [], "B2": [], "B3": [], "B4": []}
    for i in idx:
        if i < HIST - 1 or i + 1 >= len(m15):
            continue
        bars_win = m15.iloc[i - HIST + 1: i + 1]
        t = pd.Timestamp(times[i])
        entry = t + pd.Timedelta(minutes=15)
        if not bool(session_open(pd.Series([entry]))[0]):
            continue
        if float(m15["spread_usd"].iloc[i + 1]) > GATE:
            continue
        h4u = h4_utc[h4_utc["time"] <= t].tail(H4HIST)
        h4s = h4_srv[h4_srv["time"] <= t].tail(H4HIST)
        reg_win = int(RegimeEngine().compute(bars_win)["regime"].iloc[-1])
        reg_fl = int(reg_full[i])
        sig_u = signal_at(bars_win, h4u)
        sig_s = signal_at(bars_win, h4s)
        if not (sig_u or sig_s):
            continue
        def ok(sig, reg):
            return sig is not None and sig.direction > 0 and reg == TREND_UP
        if ok(sig_s, reg_win):
            counts["B1"].append(t)
        if ok(sig_u, reg_win):
            counts["B2"].append(t)
        if ok(sig_u, reg_fl):
            counts["B3"].append(t)
        if ok(sig_s, reg_fl):
            counts["B4"].append(t)
    desc = {"B1": "h4سرور + رژیم پنجره‌ای (لایو)", "B2": "h4-UTC + رژیم پنجره‌ای",
            "B3": "h4-UTC + رژیم کامل (بک‌تست)", "B4": "h4سرور + رژیم کامل"}
    for k in ("B1", "B2", "B3", "B4"):
        ts = counts[k]
        s = "، ".join(f"{pd.Timestamp(x):%m-%d %H:%M}" for x in ts[:12])
        more = f" +{len(ts)-12}…" if len(ts) > 12 else ""
        print(f"  {k} {desc[k]:32s}: {len(ts):2d} لحظه — {s}{more}")


if __name__ == "__main__":
    main()
