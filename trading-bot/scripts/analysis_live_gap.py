"""فاصلهٔ بی‌معامله‌گی — «چند روزِ بی‌معامله عادی است؟» (۱۶ سپتامبر ۲۰۲۶)

اجرا:
    py scripts/analysis_live_gap.py

پرسش: آستین طلای دموی فوروارد از ~۲۱ آگوست ۲۶ روز است بدون معامله،
در حالی که نرخ بک‌تست G6 ~۸ معامله/ماه بود. عادی است یا سیگنالِ گیرِ گیت‌ها؟

روش:
  1) همان پایپ‌لاین فاز ۷ روی دیتای ۳سالهٔ MT5 طلا (تنها نسخهٔ تأییدشده)
  2) توزیع فاصلهٔ بین ورودهای متوالی در ۳ سال — بلندترین «سکوت»ها
  3) معاملاتِ همان کانفیگ در بازهٔ لایو (۲۱ آگوست → ۱۰ سپتامبر؛
     دیتای MT5 تا ۱۰ سپتامبر موجود است — ۶ روز آخر نه)
  4) پوشش رژیم در بازهٔ لایو (رژیمِ بسته = صفرِ توضیح‌داره)
  5) نقره: فقط نرخ و بیشینهٔ سکوت (دیتا تا ۹ سپتامبر؛ لایو نقره از ~۱۲ سپتامبر)
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
from bot.regime.engine import (CHAOS, TREND_UP, RegimeEngine, session_open)

# بازهٔ لایوِ پوشش‌داده‌شده توسط دیتای موجود (طلا: MT5 تا ۱۰ سپتامبر)
W0 = pd.Timestamp("2026-08-21")
W1 = pd.Timestamp("2026-09-10")


def find(name: str) -> pathlib.Path:
    for base in ("data", "../marketdata", "marketdata"):
        p = pathlib.Path(base) / name
        if p.exists():
            return p
    raise SystemExit(f"❌ فایل {name} پیدا نشد")


def silence_report(t: pd.DataFrame, months: float) -> None:
    """توزیع فاصلهٔ بین ورودها + سکوت‌های ≥۲۰ روز در کل تاریخچه."""
    if len(t) < 2:
        print(f"  {len(t)} معامله — برای تحلیل فاصله کافی نیست")
        return
    tt = t.sort_values("entry_time")["entry_time"].reset_index(drop=True)
    days = (tt.diff().dropna().dt.total_seconds() / 86400)
    print(f"  {len(t)} معامله در {months:.0f} ماه = {len(t)/months:.1f}/ماه")
    print(f"  فاصلهٔ بین ورودها (روز): میانه {days.median():.1f} | "
          f"p90 {days.quantile(.9):.1f} | p99 {days.quantile(.99):.1f} | "
          f"بیشینه {days.max():.0f}")
    mask = days >= 20
    if mask.any():
        spans = [f"{tt[i-1].date()}→{tt[i].date()} ({days[i]:.0f}روز)"
                 for i in days.index[mask]]
        print(f"  سکوت‌های ≥۲۰ روز: {mask.sum()} مورد — " + "، ".join(spans))
    else:
        print("  سکوتِ ≥۲۰ روز: هرگز در کل تاریخچه رخ نداده")


def main() -> None:
    # ── طلا: دقیقاً کانفیگ فاز ۷ (G6) ────────────────────────────
    mt5 = load_mt5(find("xauusd_m1_mt5_3y.csv.gz"))
    duka = load_duka(find("xauusd_m1_duka.csv.gz"))
    mt5_utc = shift_mt5_to_utc(mt5, infer_offset_minutes(mt5, duka))
    m15, h4 = to_m15(mt5_utc), resample_tf(mt5_utc, "4h")
    reg = RegimeEngine().compute(m15)["regime"].to_numpy()
    sess = np.asarray(session_open(m15["time"]), dtype=bool)
    strat = GatedStrategy(V0Strategy(h4, rr=2.5), reg, sess, direction=+1)
    res = Backtester(m15, strat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, max_positions=1,
        cooldown_bars=1, slippage_usd=0.05, spread_gate_usd=0.40)).run()
    g = res.trades.assign(symbol="XAUUSD")

    print("═══ طلا (G6 — همان کانفیگ فاز ۷، ۳ سال) ═══")
    silence_report(g, 36.0)

    # چک سایزینگ لایو: $3k + ریسک ۰.۵٪ — تعداد ورود نباید عوض شود
    res_live = Backtester(m15, strat, BacktestConfig(
        start_equity=3000.0, risk_pct=0.005, max_positions=1,
        cooldown_bars=1, slippage_usd=0.05, spread_gate_usd=0.40)).run()
    same = len(res_live.trades) == len(g)
    print(f"  چک سایزینگِ لایو ($3k، ریسک ۰.۵٪): {len(res_live.trades)} معامله "
          f"{'✓ همان تعداد' if same else '⚠️ متفاوت — بررسی کن!'}")

    # بازهٔ لایو
    w = g[(g["entry_time"] >= W0) & (g["entry_time"] < W1)]
    print(f"\n── بازهٔ لایو ({W0.date()} → {W1.date()}؛ ۲۰ روزِ پوشش‌دار) ──")
    print(f"  بک‌تستِ همان کانفیگ: {len(w)} معامله | لایو: 0")
    if len(w):
        cols = [c for c in ("entry_time", "exit_time", "pnl", "reason", "lots")
                if c in w.columns]
        print(w[cols].to_string(index=False))
    gs = g.sort_values("entry_time")
    before = gs[gs["entry_time"] < W0]
    after = gs[gs["entry_time"] >= W1]
    if len(before):
        r = before.iloc[-1]
        print(f"  آخرین معاملهٔ قبل از بازه: {r['entry_time']} (pnl ${r['pnl']:+.0f})")
    if len(after):
        r = after.iloc[0]
        print(f"  اولین معاملهٔ بعد از بازه: {r['entry_time']} (pnl ${r['pnl']:+.0f})")

    # پوشش رژیم در بازهٔ لایو
    in_w = ((m15["time"] >= W0) & (m15["time"] < W1)).to_numpy()
    r_w = pd.Series(reg[in_w])
    sess_w = sess[in_w]
    print("\n── رژیم در بازهٔ لایو (سهم کندل‌های M15) ──")
    for k, v in r_w.value_counts(normalize=True).items():
        print(f"  {k}: {v:.0%}")
    m_session = r_w[sess_w]
    if len(m_session):
        print(f"  داخل سشنِ ورود (۱۲–۲۰ UTC): TREND_UP "
              f"{(m_session == TREND_UP).mean():.0%} | "
              f"CHAOS {(m_session == CHAOS).mean():.0%}")

    # ماه‌به‌ماه ۲۰۲۶ — آیا نرخِ اخیر از میانگین ۳ساله کمتر شده؟
    mo = g.assign(m=g["entry_time"].dt.to_period("M")).groupby("m").size()
    recent = mo[mo.index >= pd.Period("2026-01", freq="M")]
    print("\n── معاملات ماهانهٔ طلا در ۲۰۲۶ ──")
    for k, v in recent.items():
        print(f"  {k}: {v}")

    # ── نقره: کانفیگ فاز ۷ (اسپرد ثابت $0.018) ───────────────────
    print("\n═══ نقره (فاز ۷ — اسپرد ثابت $0.018، ۳ سال) ═══")
    sil = load_duka(find("xagusd_m1_duka.csv.gz"))
    s15, s4 = to_m15(sil), resample_tf(sil, "4h")
    sreg = RegimeEngine().compute(s15)["regime"].to_numpy()
    ssess = np.asarray(session_open(s15["time"]), dtype=bool)
    m = s15.copy()
    m["spread_usd"] = 0.018
    sstrat = GatedStrategy(V0Strategy(s4, rr=2.5, sl_pad=0.01093),
                           sreg, ssess, direction=+1)
    sres = Backtester(m, sstrat, BacktestConfig(
        start_equity=30000.0, fixed_lots=0.01, contract_oz=5000.0,
        max_positions=1, cooldown_bars=1, slippage_usd=0.00109,
        spread_gate_usd=None)).run()
    s = sres.trades.assign(symbol="XAGUSD")
    silence_report(s, 36.0)
    smo = s.assign(m=s["entry_time"].dt.to_period("M")).groupby("m").size()
    srecent = smo[smo.index >= pd.Period("2026-06", freq="M")]
    print("  ماه‌های اخیر: " +
          "، ".join(f"{k}={v}" for k, v in srecent.items()))


if __name__ == "__main__":
    main()
