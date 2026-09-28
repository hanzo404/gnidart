"""آزمون نبودِ آینده‌نگری در آشکارساز ساختار.

قانون: اگر دادهٔ آینده را عوض کنیم، سیگنال‌های *قبل* از آن لحظه نباید تغییر کنند.
این آزمون «قطع‌کردن آینده» (causal truncation) دقیق‌ترین راه تشخیص آینده‌نگری است.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ictlab.structure import StructureParams, atr, detect_sweeps, swings, liquidity_pools


def make(n=4000, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.03, n)
    close = 100 + np.cumsum(ret)
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.02, n))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.02, n))
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"time": t, "open": open_, "high": hi, "low": lo, "close": close})


def test_sweeps_are_causal() -> None:
    df = make()
    p = StructureParams()
    a = atr(df, 14)
    full = detect_sweeps(df, p, a)

    cut = 2500
    df2 = df.iloc[:cut].copy()
    a2 = atr(df2, 14)
    trunc = detect_sweeps(df2, p, a2)

    key = lambda d: [(r.time, r.dir, round(r.level, 6), round(r.extreme, 6))
                     for r in d.itertuples() if r.time < df2.time.iloc[-1]]
    f, t_ = key(full), key(trunc)
    assert f == t_, (f"سیگنال‌ها پس از حذف آینده تغییر کردند: "
                     f"{len(f)} در برابر {len(t_)}")
    print(f"  [OK] جاروب‌ها علّی‌اند: {len(f):,} سیگنال، یکسان پس از قطع آینده")


def test_pools_are_causal() -> None:
    df = make(seed=2)
    p = StructureParams()
    typ, _, prc = swings(df, p.swing_k)
    tol = atr(df, 14) * p.pool_tol_atr
    full = liquidity_pools(typ, prc, tol)

    cut = 2200
    typ2, _, prc2 = swings(df.iloc[:cut], p.swing_k)
    tol2 = atr(df.iloc[:cut], 14) * p.pool_tol_atr
    trunc = liquidity_pools(typ2, prc2, tol2)

    assert np.allclose(full[:cut, 1], trunc[:, 1], equal_nan=True), "سطوح پایین آینده‌نگرانه‌اند"
    assert np.allclose(full[:cut, 0], trunc[:, 0], equal_nan=True), "سطوح بالا آینده‌نگرانه‌اند"
    print(f"  [OK] استخرهای نقدشوندگی علّی‌اند: {int(np.isfinite(full[:cut,0]).sum())} سقف، "
          f"{int(np.isfinite(full[:cut,1]).sum())} کف تا لحظهٔ قطع")


def test_swings_confirm_after_k_bars() -> None:
    """سوئینگ در کندل i ساخته و در کندل i+k تأیید می‌شود؛ نه زودتر."""
    df = make(seed=3)
    k = 3
    typ, idx, prc = swings(df, k)
    n_hi = n_lo = 0
    for c in range(len(df)):
        if typ[c] == 0:
            continue
        s = idx[c]                      # اندیس واقعیِ کندلِ سوئینگ
        assert c == s + k, f"سوئینگ زودتر از {k} کندل بعد تأیید شده است"
        if typ[c] == 1:
            w = df.high.values[s - k:s + k + 1]
            assert df.high.values[s] == w.max(), "سوئینگ سقف درست نیست"
            n_hi += 1
        else:
            w = df.low.values[s - k:s + k + 1]
            assert df.low.values[s] == w.min(), "سوئینگ کف درست نیست"
            n_lo += 1
    print(f"  [OK] تعریف سوئینگ: {n_hi:,} سقف و {n_lo:,} کف، همگی با تأخیر {k} کندل")


if __name__ == "__main__":
    print("آزمون‌های علّی‌بودن ساختار:")
    test_swings_confirm_after_k_bars()
    test_pools_are_causal()
    test_sweeps_are_causal()
    print("همه سبز ✅")
