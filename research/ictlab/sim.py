"""شبیه‌ساز رویدادمحور با هزینهٔ واقعی.

قواعد سخت‌گیرانهٔ پرهیز از خودفریبی (این‌ها جایی نیست که شکست بخوریم):
  ۱) سیگنال در کندل t تأیید می‌شود ⇒ اولین فرصت ورود، کندل t+1 است.
  ۲) اگر در یک کندل هم استاپ و هم هدف لمس شود، **بدبینانه** استاپ فرض می‌شود.
  ۳) ورود لیمیت فقط وقتی پر می‌شود که کندل واقعاً آن قیمت را لمس کند.
  ۴) اسلیپیج و اسپرد در هر ورود/خروج کسر می‌شود.
  ۵) در هر لحظه فقط یک پوزیشن باز (بدون انباشت و بدون مارتینگل).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Spec


@dataclass
class SimParams:
    rr: float = 2.0                # نسبت هدف به ریسک
    stop_mode: str = "sweep"       # "sweep" (پشت extremum جاروب) | "atr" (n×ATR)
    stop_atr_mult: float = 1.0     # وقتی stop_mode="atr"
    stop_buffer_atr: float = 0.10  # بافر پشت extremum، بر حسب ATR
    entry_offset: float = 0.5      # ورود در چه نسبتی از ناحیه: 0=لبه نزدیک، 1=لبه دور
    expiry_bars: int = 6           # انقضای سفارش لیمیت (واحد: کندل سیگنال)
    max_hold_bars: int = 90        # حداکثر نگهداری (کندل M1)
    min_stop_spread: float = 3.0   # حداقل فاصلهٔ استاپ بر حسب «برابرِ اسپرد»
    time_stop_r: float = 0.0       # اگر >0، خروج زمانی در اکویتی مثبت
    allow_short: bool = True
    allow_long: bool = True


@dataclass
class Trade:
    time: pd.Timestamp
    dir: int
    entry: float
    stop: float
    target: float
    exit: float
    r: float
    reason: str
    bars_held: int
    mfe: float
    mae: float
    # قیمتِ لیمیتِ دستور، بدون اسپرد و اسلیپیج. اگر قیمت «از استاپ پر» شود
    # (گپ), مقدارِ entry پرشده می‌تواند به آن‌سوی استاپ بیفتد؛ این فیلد
    # مرجعِ بدون‌ابهامِ جهت است.
    entry_signal: float = float("nan")


def simulate(
    m1: pd.DataFrame,
    signals: pd.DataFrame,
    spec: Spec,
    p: SimParams,
    slip_mult: float = 1.0,
    spread_col: np.ndarray | None = None,
) -> pd.DataFrame:
    """اجرای سیگنال‌ها روی کندل M1. signals باید ستون‌های زیر را داشته باشد:
    time, dir, entry (قیمت لیمیت), stop (قیمت خام، بدون بافر), tag (اختیاری)
    """
    if signals.empty:
        return pd.DataFrame()

    mtime = m1.time.values.astype("datetime64[ns]")
    m_open = m1.open.values
    m_high = m1.high.values
    m_low = m1.low.values
    m_close = m1.close.values
    # اسپردِ واقعی هر کندل اگر داده شود؛ وگرنه مقدار ثابتِ مشخصات نماد.
    # این برای طلا حیاتی است: اسپرد از ۰.۱۴ در ساعت‌های آرام تا بیش از ۱.۰
    # در لحظه‌های بی‌نقدشوندگی نوسان می‌کند.
    spr_arr = (np.asarray(spread_col, dtype=np.float64) if spread_col is not None
               else None)
    spread = spec.spread_pts * spec.point
    slip = spec.slippage_pts * spec.point * slip_mult

    # نگاشت زمان → اندیس (جست‌وجوی دودویی)
    sig_time = signals.time.values.astype("datetime64[ns]")
    start = np.searchsorted(mtime, sig_time, side="left") + 1  # کندل بعدی: بدون آینده‌نگری

    trades: list[Trade] = []
    busy_until = 0  # اندیس M1 که تا آن موقع درگیر هستیم

    for i, (st, direction, entry, stop, tag) in enumerate(
        zip(start, signals.dir.values, signals.entry.values, signals.stop.values,
            signals.tag.values if "tag" in signals.columns else np.array([""] * len(signals)))
    ):
        if direction == 1 and not p.allow_long:
            continue
        if direction == -1 and not p.allow_short:
            continue
        if st >= len(m_open) or st < busy_until:
            continue

        spr_here = float(spr_arr[st]) if spr_arr is not None else spread
        risk = abs(entry - stop)
        if risk < p.min_stop_spread * spr_here or risk <= 0:
            continue
        # جهتِ استاپ باید با جهتِ معامله سازگار باشد (استاپ پشت قیمت)
        if direction * (entry - stop) <= 0:
            continue
        # «لیمیت» باید واقعاً لیمیت باشد: خرید زیرِ قیمت، فروش بالای قیمت.
        # اگر سطح از قبل رد شده باشد، ستاپ گذشته است و نباید وارد شویم.
        ref = m_close[st - 1]
        if (direction == 1 and entry >= ref) or (direction == -1 and entry <= ref):
            continue

        expiry_idx = min(st + p.expiry_bars * 5, len(m_open))  # پنجرهٔ انقضا به دقیقه

        # ── مرحلهٔ ۱: پر شدن سفارش لیمیت (کاملاً متقارن برای خرید/فروش)
        fill = -1
        fill_price = np.nan
        for j in range(st, expiry_idx):
            o, hi, lo = m_open[j], m_high[j], m_low[j]
            if direction == 1:                       # خرید لیمیت زیرِ بازار
                if lo <= entry:
                    fill = j
                    fill_price = o if o <= entry else entry
                    break
            else:                                    # فروش لیمیت بالای بازار
                if hi >= entry:
                    fill = j
                    fill_price = o if o >= entry else entry
                    break
        if fill < 0:
            continue
        # کندل‌ها بر پایهٔ BID هستند ⇒ خرید در ASK پر می‌شود، فروش در BID.
        fill_spread = float(spr_arr[fill]) if spr_arr is not None else spread
        fill_price += (fill_spread + slip) if direction == 1 else -slip

        # فاصلهٔ واقعی ریسک پس از اسلیپیج
        real_risk = abs(fill_price - stop)
        if real_risk <= 0:
            continue
        target = fill_price + direction * p.rr * real_risk
        # بستن معامله هم هزینه دارد: فروش در BID رایگان است، خرید در ASK گران
        exit_cost = (-slip if direction == 1 else (fill_spread + slip))

        # ── مرحلهٔ ۲: مدیریت خروج
        exit_price, reason, bars_held = np.nan, "", 0
        mfe = mae = 0.0
        last = min(fill + p.max_hold_bars, len(m_open))
        for j in range(fill, last):
            o, hi, lo = m_open[j], m_high[j], m_low[j]
            bars_held = j - fill + 1
            fav = (hi - fill_price) * direction
            adv = (lo - fill_price) * direction
            mfe = max(mfe, fav / real_risk)
            mae = min(mae, adv / real_risk)
            # لمس‌ها به‌درستی تعریف می‌شوند و «اول استاپ» بدبینانه است
            if direction == 1:
                stop_hit, tp_hit = (lo <= stop), (hi >= target)
            else:
                stop_hit, tp_hit = (hi >= stop), (lo <= target)
            if stop_hit:
                exit_price = (o if (o <= stop) else stop) + exit_cost
                reason = "sl"
                break
            if tp_hit:
                exit_price = (o if (o >= target) else target) + exit_cost
                reason = "tp"
                break
        if not reason:
            exit_price = m_open[last] + exit_cost
            reason = "time"
        busy_until = j + 1

        pnl = (exit_price - fill_price) * direction
        trades.append(
            Trade(
                time=pd.Timestamp(mtime[fill]), dir=direction, entry=fill_price, stop=stop,
                target=target, exit=exit_price, r=pnl / real_risk, reason=reason,
                bars_held=bars_held, mfe=mfe, mae=mae,
                # قیمتِ لیمیتِ دستور (بدون اسپرد/اسلیپیج). اگر قیمت از
                # استاپ «پر» شود (گپ)، entry پرشده می‌تواند به آن�� طرف دیگر
                # بیفتد؛ entry_signal مرجعِ بدون‌ابهامِ جهت است.
                entry_signal=entry,
            )
        )
    if not trades:
        return pd.DataFrame()
    out = pd.DataFrame([t.__dict__ for t in trades])
    out["symbol"] = spec.symbol
    return out
