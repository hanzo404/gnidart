"""تست واحدهای اسپرد.

این یکی از خطرناک‌ترین باگ‌هایی بود که در این پروژه رخ داد: ستون spread در
خروجی MT5 بر حسب «نقطه» است (۱۴ یعنی ۰٫۱۴ دلار برای point=0.01)، ولی اگر
مستقیم به‌عنوان «واحد قیمت» استفاده شود، هزینه ۱۰۰ برابر واقع می‌شود و
نتیجهٔ آزمایش کاملاً وارونه می‌شود — و چون خطا به نفع رد کردنِ استراتژی
جواب نمی‌دهد، خودِ نتیجه هم بی‌صدا غلط می‌شود.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ictlab.data import Spec, load_m1


def _write(tmp: str, spread_col: bool, spread_value) -> str:
    n = 300
    p = 1950 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, n))
    d = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": p, "high": p + 0.5, "low": p - 0.5, "close": p,
        "tick_volume": 10,
    })
    if spread_col:
        d["spread"] = spread_value
    path = os.path.join(tmp, "t.csv.gz")
    d.to_csv(path, index=False, compression="gzip")
    return path


def test_mt5_spread_column_is_converted_to_price_units():
    """۱۴ در فایل MT5 یعنی ۰٫۱۴ دلار، نه ۱۴ دلار."""
    spec = Spec("T", None, point=0.01, spread_pts=14.0, slippage_pts=3.0)
    with tempfile.TemporaryDirectory() as tmp:
        m = load_m1(spec, _write(tmp, True, 14))
        assert abs(np.median(m.spread.values) - 0.14) < 1e-9, \
            f"اسپرد باید 0.14 باشد، نه {np.median(m.spread.values)}"


def test_missing_spread_column_uses_spec_in_price_units():
    spec = Spec("T", None, point=0.01, spread_pts=14.0, slippage_pts=3.0)
    with tempfile.TemporaryDirectory() as tmp:
        m = load_m1(spec, _write(tmp, False, None))
        assert abs(np.median(m.spread.values) - 0.14) < 1e-9


def test_spread_is_plausible_relative_to_bar_range():
    """اسپرد نباید از دامنهٔ معمول کندل بزرگ‌تر باشد.

    این تست به‌صورت غیرمستقیم از خطای واحد محافظت می‌کند: اگر واحدها
    دوباره خراب شوند، اسپرد از کل دامنهٔ M1 بزرگ‌تر می‌شود.
    """
    spec = Spec("T", None, point=0.01, spread_pts=14.0, slippage_pts=3.0)
    with tempfile.TemporaryDirectory() as tmp:
        m = load_m1(spec, _write(tmp, True, 14))
        rng = (m.high - m.low)
        ratio = np.median(m.spread.values) / np.median(rng)
        assert 0.01 < ratio < 0.5, f"نسبت اسپرد به دامنه غیرمنطقی است: {ratio:.3f}"


def test_cost_actually_reduces_expectancy():
    """هزینهٔ بیشتر باید حتماً نتیجه را بدتر کند — تستِ یکنواختی.

    اگر واحدهای اسپرد خراب شود، این یکنواختی می‌شکند.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ictlab.sim import SimParams, simulate

    rng = np.random.default_rng(5)
    n = 30000
    p = 1950 + np.cumsum(rng.normal(0, 0.3, n))
    m = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": p, "high": p + 0.4, "low": p - 0.4, "close": p,
        "volume": 1, "spread": 0.14,
    })
    # ورودهای تصادفی: انتظار داریم هزیجه آن‌ها را بخورد
    # ورودهای تصادفی در جایی که واقعاً لیمیتِ معتبر است: برای خرید زیرِ
    # بازار، برای فروش بالای بازار. (ورود روی قیمت بسته‌شدن «بازار» است و
    # شبیه‌ساز آن را عمداً رد می‌کند.)
    # فاصلهٔ کافی بین سیگنال‌ها تا قفلِ «یک معامله همزمان» کمترین حذف را بکند
    idx = np.arange(100, n - 200, 40)
    side = rng.choice([1, -1], len(idx))
    entry = np.where(side == 1, m.close.values[idx] - 0.30, m.close.values[idx] + 0.30)
    sig = pd.DataFrame({
        "time": m.time.values[idx], "dir": side, "entry": entry,
        "stop": np.where(side == 1, entry - 0.80, entry + 0.80),
    })
    thin = Spec("T", None, point=0.01, spread_pts=14.0, slippage_pts=3.0)
    fat = Spec("T", None, point=0.01, spread_pts=28.0, slippage_pts=6.0)
    a = simulate(m, sig, thin, SimParams(rr=1.5, max_hold_bars=30, min_stop_spread=0.0))
    b = simulate(m, sig, fat, SimParams(rr=1.5, max_hold_bars=30, min_stop_spread=0.0))
    assert len(a) > 200 and len(b) > 200, "ورودهای تصادفی باید اجرا شوند"
    assert b.r.mean() < a.r.mean(), "هزینهٔ بیشتر باید نتیجه را بدتر کند"
