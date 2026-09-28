"""آزمون‌های صحت شبیه‌ساز — سنگربانِ ضدخودفریبی.

قاعده: تا وقتی این آزمون‌ها سبز نشده‌اند، هیچ عددی از بک‌تست جدی گرفته نمی‌شود.
هر باگی که در گذشتهٔ این پروژه «استراتژی شگفت‌انگیز» ساخته، از همین دسته بوده است.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ictlab.data import Spec
from ictlab.sim import SimParams, simulate

SPEC = Spec("TEST", "none", point=0.01, spread_pts=1.0, slippage_pts=0.0)


def synth(n=200_000, seed=3) -> pd.DataFrame:
    """بازار مصنوعی: قدم‌زدن تصادفی بدون روند."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.02, n)          # گام تصادفی
    close = 100 + np.cumsum(ret)
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.01, n))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.01, n))
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": t, "open": open_, "high": hi, "low": lo, "close": close,
                         "volume": 1.0, "spread": 0.01})


def test_no_free_lunch_random_entries() -> None:
    """ورود تصادفی روی بازار بدون روند ⇒ امید ریاضی نزدیک صفر، WR نزدیک ۵۰٪."""
    df = synth()
    rng = np.random.default_rng(99)
    # شبکهٔ منظم با فاصلهٔ ۹۰ کندل (> حداکثر نگهداری) تا هیچ سیگنالی
    # به‌خاطر «یک پوزیشن هم‌زمان» حذف نشود
    n_sig = 2000
    bars = 10 + np.arange(n_sig) * 90
    dirs = rng.choice([1, -1], n_sig)
    entry = df.close.values[bars] - dirs * 0.05     # لیمیتِ واقعی
    sig = pd.DataFrame({"time": df.time.values[bars], "dir": dirs, "entry": entry,
                        "stop": entry - dirs * 0.5,  # استاپ پشت ورود
                        "tag": ["rnd"] * n_sig})
    tr = simulate(df, sig, SPEC, SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=60))
    assert len(tr) > 1200, f"پرشدن سفارش‌ها غیرمنتظره کم است: {len(tr)}"
    e = tr.r.mean()
    wr = 100 * (tr.r > 0).mean()
    se = tr.r.std() / np.sqrt(len(tr))
    print(f"  [OK] تصادفی: n={len(tr):,}  E[R]={e:+.4f} (t={e/se:+.2f})  WR={wr:.1f}%  PF={tr[tr.r>0].r.sum()/-tr[tr.r<0].r.sum():.2f}")
    # «ناهار رایگان» وجود ندارد: امید ریاضیِ تصادفی نباید مثبت و معنادار باشد.
    # (انتظار داریم منفی باشد، چون هزینهٔ معامله پرداخت می‌شود.)
    assert e / se < 2.0, f"شبیه‌ساز سوگیری دارد: امید ریاضیِ تصادفی {e:+.4f} (t={e/se:+.2f})"
    assert e < 0, "ورود تصادفی باید به‌خاطر هزینه ضرر بدهد"


def test_short_tp_requires_downward_move() -> None:
    """بازگشتِ باگ: در معاملهٔ فروش، قیمتِ بالا رفتن هرگز نباید «هدف» باشد."""
    df = synth(n=20_000, seed=5)
    b = 1000
    sig = pd.DataFrame({"time": [df.time.values[b]], "dir": [-1],
                        "entry": [df.close.values[b] + 0.30],   # فروش، بالاتر از قیمت
                        "stop": [df.close.values[b] + 0.80],
                        "tag": ["t"]})
    tr = simulate(df, sig, SPEC, SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=30))
    if len(tr):
        assert not (tr.reason == "tp").any(), "فروش بدون حرکت نزولی به هدف رسیده — باگ زنده است!"
    print("  [OK] فروش: هدف فقط با حرکت نزولی لمس می‌شود")


def test_long_short_symmetry() -> None:
    """روی همان سیگنال‌های تصادفی، خرید و فروش نباید سوگیری سیستماتیک داشته باشند."""
    df = synth(seed=8)
    rng = np.random.default_rng(4)
    bars = np.sort(rng.choice(np.arange(10, len(df) - 200), 2500, replace=False))
    out = {}
    for d in (1, -1):
        entry = df.close.values[bars] - d * 0.05
        s = pd.DataFrame({"time": df.time.values[bars], "dir": d, "entry": entry,
                          "stop": entry - d * 0.4, "tag": ["x"] * len(bars)})
        tr = simulate(df, s, SPEC, SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=60))
        out[d] = tr.r.mean()
        print(f"  [{'OK' if d==1 else '  '}] dir={d:+d}: E[R]={out[d]:+.4f}")
    diff = abs(out[1] - out[-1])
    assert diff < 0.05, f"عدم تقارن خرید/فروش: {diff:.4f}"


def test_cost_is_charged() -> None:
    """هزینه باید حتماً کسر شود: با همان سیگنال، افزودنِ اسلیپیج E[R] را کم کند."""
    df = synth(n=60_000, seed=12)
    rng = np.random.default_rng(21)
    bars = np.sort(rng.choice(np.arange(10, len(df) - 200), 1500, replace=False))
    dirs = rng.choice([1, -1], len(bars))
    entry = df.close.values[bars] - dirs * 0.05
    sig = pd.DataFrame({"time": df.time.values[bars], "dir": dirs, "entry": entry,
                        "stop": entry - dirs * 0.5, "tag": ["x"] * len(bars)})
    free = simulate(df, sig, Spec("T", "n", 0.01, 0.01, 0.0),
                    SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=60)).r.mean()
    costed = simulate(df, sig, Spec("T", "n", 0.01, 1.0, 2.0),
                      SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=60)).r.mean()
    print(f"  [OK] هزینه: بدون هزینه E[R]={free:+.4f} → با هزینه E[R]={costed:+.4f}")
    assert costed < free, "هزینهٔ معامله اعمال نشده است!"


def test_spread_is_charged() -> None:
    """اسپرد (BID/ASK) باید در ورود و خروج کسر شود — نه فقط اسلیپیج."""
    df = synth(n=40_000, seed=44)
    rng = np.random.default_rng(77)
    bars = 100 + np.arange(800) * 40
    dirs = rng.choice([1, -1], len(bars))
    entry = df.close.values[bars] - dirs * 0.05
    sig = pd.DataFrame({"time": df.time.values[bars], "dir": dirs, "entry": entry,
                        "stop": entry - dirs * 0.4, "tag": ["x"] * len(bars)})
    free = simulate(df, sig, Spec("T", "n", 0.01, 0.0, 0.0),
                    SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=20)).r.mean()
    wide = simulate(df, sig, Spec("T", "n", 0.01, 3.0, 0.0),
                    SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=20)).r.mean()
    print(f"  [OK] اسپرد: بدون اسپرد E[R]={free:+.4f} → با اسپرد ۳ واحدی E[R]={wide:+.4f}")
    # هزینهٔ نظری ≈ spread/risk = 0.03/0.4 = 0.075R در هر معامله
    assert wide < free - 0.03, "اسپرد کسر نشده است!"


def test_no_lookahead_fill() -> None:
    """سیگنالِ ثبت‌شده در کندل t نباید در همان کندل پر شود."""
    df = synth(n=50_000, seed=17)
    rng = np.random.default_rng(31)
    bars = 100 + np.arange(1000) * 40
    dirs = rng.choice([1, -1], len(bars))
    entry = df.close.values[bars] - dirs * 0.05
    sig = pd.DataFrame({"time": df.time.values[bars], "dir": dirs, "entry": entry,
                        "stop": entry - dirs * 0.4, "tag": ["x"] * len(bars)})
    tr = simulate(df, sig, SPEC, SimParams(rr=2.0, min_stop_spread=0.0, max_hold_bars=20))
    assert len(tr) > 0
    first_sig = pd.Timestamp(sig.time.values[0])
    assert tr.time.min() > first_sig, "معامله در همان کندلِ سیگنال پر شده — آینده‌نگری!"
    # هر معامله باید بعد از آخرین سیگنالِ پیش از خودش باشد
    sig_sorted = np.sort(sig.time.values)
    for t in tr.time.values[:200]:
        j = np.searchsorted(sig_sorted, t, side="right") - 1
        if j >= 0:
            assert t > sig_sorted[j], "پر شدن معامله پیش از سیگنالش!"
    print(f"  [OK] بدون آینده‌نگری: {len(tr):,} معامله، همگی پس از لحظهٔ سیگنال")


if __name__ == "__main__":
    print("آزمون‌های صحت شبیه‌ساز:")
    test_no_free_lunch_random_entries()
    test_short_tp_requires_downward_move()
    test_long_short_symmetry()
    test_cost_is_charged()
    test_spread_is_charged()
    test_no_lookahead_fill()
    print("همهٔ آزمون‌ها سبز ✅")


def test_variable_spread_matches_constant():
    """مسیر اسپردِ متغیر باید وقتی آرایه ثابت است، دقیقاً مثل مسیر قدیمی باشد."""
    import numpy as np
    import pandas as pd
    from ictlab.data import Spec
    from ictlab.sim import SimParams, simulate

    spec = Spec("T", None, point=0.01, spread_pts=14.0, slippage_pts=3.0)
    rng = np.random.default_rng(7)
    n = 3000
    m1 = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": 2000 + np.cumsum(rng.normal(0, 0.2, n)),
    })
    m1["high"] = m1.open + 0.6
    m1["low"] = m1.open - 0.6
    m1["close"] = m1.open + rng.normal(0, 0.1, n)
    m1["volume"] = 1
    m1["spread"] = 14.0
    sig = pd.DataFrame({
        "time": m1.time.iloc[100:200].values, "dir": 1, "stop": m1.low.iloc[100:200].values - 0.3,
        "entry": m1.close.iloc[100:200].values - 0.1, "bar": np.arange(100, 200),
    })
    p = SimParams(rr=1.5, max_hold_bars=30, min_stop_spread=3.0)
    a = simulate(m1, sig, spec, p)
    b = simulate(m1, sig, spec, p, spread_col=np.full(n, 0.14))
    assert len(a) == len(b)
    assert np.allclose(a.r.values, b.r.values)


def test_zero_spread_means_no_cost():
    """با اسپرد صفر، هزینهٔ ورود و خروج باید دقیقاً صفر باشد (بجز اسلیپیج)."""
    import numpy as np
    import pandas as pd
    from ictlab.data import Spec
    from ictlab.sim import SimParams, simulate

    spec = Spec("T", None, point=0.01, spread_pts=0.0, slippage_pts=0.0)
    m1 = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=400, freq="1min"),
        "open": 2000.0,
    })
    m1["high"], m1["low"], m1["close"] = 2001.0, 1999.0, 2000.0
    m1["volume"], m1["spread"] = 1, 0.0
    sig = pd.DataFrame({"time": m1.time.iloc[50:60].values, "dir": 1, "stop": 1998.0,
                        "entry": 1999.5, "bar": np.arange(50, 60)})
    tr = simulate(m1, sig, spec, SimParams(rr=1.5, max_hold_bars=20, min_stop_spread=0.0))
    assert len(tr) > 0
    # ورود دقیقاً روی entry و خروج دقیقاً روی close
    assert np.allclose(tr.entry.values, 1999.5)
