"""آزمایش ۱۰ — سنجشِ منصفانهٔ یک EA آمادهٔ بیرونی.

موضوع: کدی که کاربر داده است (`PriceAction_Regime_EA`):
    فیلتر روند با EMA50 + شرط «کندل با بدنهٔ قوی» + ورود بازار در کندل بعدی
    با حد ضرر و حد سودِ ثابتِ نقطه‌ای (پیش‌فرض ۳۰۰ / ۶۰۰).

روش: منطق آن **دقیقاً** بازتولید و با همان شبیه‌ساز، همان اسپرد واقعی و
همان آمارِ پروژه روی هر سه بازار اجرا می‌شود. نه سخت‌گیری و نه laxity؛
فقط یکسان‌سازی.

این کد در دستهٔ «EAهای آماده/خریداری‌شده» است که در تاریخچهٔ پروژه رد شده
بود. این آزمایش برای اثبات یا ردِ آن است، نه برای توجیه.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ictlab.data import SPECS, load_m1
from ictlab.structure import atr
from ictlab.sim import Trade
from ictlab.stats import metrics, block_bootstrap_tstat, monthly_table

pd.set_option("display.width", 250)

def ema(close: np.ndarray, period: int) -> np.ndarray:
    """EMA دقیقاً مثل iMA(..., MODE_EMA, PRICE_CLOSE) در MQL5."""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values


def third_party_signals(m1: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    """شرطِ کدِ داده‌شده، بدون هیچ افزوده یا کاستن.

    isUptrend      : close1 > ema1
    isBullishPinBar: close1 > open1 و Wick بالایی < بدنه
    ورود: خرید/فروش بازار در کندل بعدی
    """
    o, h, l, c = (m1.open.values, m1.high.values, m1.low.values, m1.close.values)
    e = ema(c, period)
    body_up = c - o
    upper_wick = h - c
    body_dn = o - c
    lower_wick = c - l
    buy = (c > e) & (body_up > 0) & (upper_wick < body_up)
    sell = (c < e) & (body_dn > 0) & (lower_wick < body_dn)
    t = m1.time.values
    sig = pd.DataFrame({
        "time": np.concatenate([t[:-1][buy[:-1]], t[:-1][sell[:-1]]]),
        "dir": np.concatenate([np.ones(buy[:-1].sum(), int), -np.ones(sell[:-1].sum(), int)]),
    })
    return sig.sort_values("time").reset_index(drop=True)


def run_market(m1: pd.DataFrame, sig: pd.DataFrame, spec, sl_price: float, tp_price: float,
               max_hold: int = 60, spread_col=None) -> pd.DataFrame:
    """ورود بازار در کندل بعدی + SL/TP ثابت، دقیقاً مثل EA.

    هزینه‌ها هم‌راستا با شبیه‌ساز اصلی: خرید = قیمت + اسپرد + اسلیپیج،
    فروش = قیمت − اسلیپیج، و خروج هم هزینه دارد.
    """
    if sig.empty:
        return pd.DataFrame()
    mtime = m1.time.values.astype("datetime64[ns]")
    o, h, l = m1.open.values, m1.high.values, m1.low.values
    spread = spec.spread_pts * spec.point
    slip = spec.slippage_pts * spec.point
    spr_arr = np.asarray(spread_col, float) if spread_col is not None else None
    start = np.searchsorted(mtime, sig.time.values.astype("datetime64[ns]"), side="left") + 1

    trades, busy = [], 0
    for st, d in zip(start, sig.dir.values):
        if st >= len(o) or st < busy:
            continue
        spr = float(spr_arr[st]) if spr_arr is not None else spread
        entry = o[st] + (spr + slip) if d == 1 else o[st] - slip
        stop = entry - d * sl_price
        target = entry + d * tp_price
        exit_price, reason, j = np.nan, None, None
        for k in range(st, min(st + max_hold, len(o))):
            stop_hit = (l[k] <= stop) if d == 1 else (h[k] >= stop)
            tp_hit = (h[k] >= target) if d == 1 else (l[k] <= target)
            if stop_hit:      # بدبینانه: استاپ اول
                exit_price = (o[k] if ((o[k] <= stop) if d == 1 else (o[k] >= stop)) else stop)
                exit_price += -slip if d == 1 else (spr + slip)
                reason = "sl"
                j = k
                break
            if tp_hit:
                exit_price = (o[k] if ((o[k] >= target) if d == 1 else (o[k] <= target)) else target)
                exit_price += -slip if d == 1 else (spr + slip)
                reason = "tp"
                j = k
                break
        if j is None:
            j = min(st + max_hold - 1, len(o) - 1)
            exit_price = o[j] + (-slip if d == 1 else (spr + slip))
            reason = "time"
        busy = j + 1
        risk = abs(entry - stop)
        trades.append(Trade(
            time=pd.Timestamp(mtime[st]), dir=int(d), entry=entry, stop=stop, target=target,
            exit=exit_price, r=(exit_price - entry) * d / risk, reason=reason,
            bars_held=j - st + 1, mfe=0.0, mae=0.0))
    return pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()


def report(name: str, tr: pd.DataFrame) -> None:
    if tr is None or len(tr) < 120:
        print(f"  {name}: n={0 if tr is None else len(tr)}  (برای قضاوت کم است)")
        return
    m, bs = metrics(tr), block_bootstrap_tstat(tr)
    mt = monthly_table(tr)
    print(f"  {name:34s} n={m['n']:6,}  WR={m['win_rate']:4.1f}%  E[R]={m['expectancy_r']:+.3f}  "
          f"PF={m['profit_factor']:5.2f}  t={bs['t']:+6.2f}  "
          f"CI=[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]  DD={m['max_dd_pct']:6.1f}%  "
          f"ماهِ مثبت={100 * (mt.r_sum > 0).mean():3.0f}%")


def main() -> None:
    for sym in ("US30", "NAS100", "XAUUSD"):
        spec = SPECS[sym]
        m1 = load_m1(spec)
        a = atr(m1, 14)
        spr = m1.spread.values if "spread" in m1.columns else None
        rng = (m1.high - m1.low)
        sig = third_party_signals(m1)
        print("=" * 118)
        print(f"{sym}  —  {m1.time.min().date()} تا {m1.time.max().date()}، "
              f"{len(m1):,} کندل، اسپرد {float(np.median(spr)):.4f}")
        print(f"  سیگنالِ خام: {len(sig):,} ({100*len(sig)/len(m1):.2f}٪ کندل‌ها)  |  "
              f"دامنهٔ M1 میانه={rng.median():.2f}  ATR={np.median(a):.2f}")

        print("\n  ── پیش‌فرضِ خودِ کد: SL=300pt، TP=600pt (RR=2) ──")
        sl = 300 * spec.point
        report(f"SL={sl:.2f} TP={sl*2:.2f}", run_market(m1, sig, spec, sl, sl * 2, spread_col=spr))
        med_spr = float(np.median(spr)) if spr is not None else spec.spread_pts * spec.point
        print(f"     ⇒ نسبتِ استاپ به ATR: {sl/np.median(a):.2f} برابر  "
              f"(اسپرد {med_spr:.4f} = {100*med_spr/sl:.1f}٪ از استاپ)")

        print("\n  ── جابه‌جایی مقیاس: استاپ از ۰٫۵ تا ۴ برابرِ ATR ──")
        for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            slp = k * float(np.median(a))
            report(f"SL={k}×ATR ({slp:.1f}) RR=2", run_market(m1, sig, spec, slp, slp * 2, spread_col=spr))

        print("\n  ── مقایسهٔ منصفانه با استراتژی خودمان: RR=1.5 ──")
        for k in (1.0, 2.0):
            slp = k * float(np.median(a))
            report(f"SL={k}×ATR RR=1.5", run_market(m1, sig, spec, slp, slp * 1.5, spread_col=spr))

        print("\n  ── آیا فیلتر EMA اصلاً کاری می‌کند؟ (بدون EMA: فقط کندل قوی) ──")
        o, h, l, c = m1.open.values, m1.high.values, m1.low.values, m1.close.values
        up = (c - o) > 0; dn = (o - c) > 0
        t = m1.time.values
        raw = pd.DataFrame({
            "time": np.concatenate([t[:-1][(up & (h - c < c - o))[:-1]],
                                    t[:-1][(dn & (c - l < o - c))[:-1]]]),
            "dir": np.concatenate([np.ones((up & (h - c < c - o))[:-1].sum(), int),
                                   -np.ones((dn & (c - l < o - c))[:-1].sum(), int)]),
        }).sort_values("time").reset_index(drop=True)
        slp = float(np.median(a))
        report("بدون فیلتر EMA، SL=ATR", run_market(m1, raw, spec, slp, slp * 2, spread_col=spr))
        print()


if __name__ == "__main__":
    main()
