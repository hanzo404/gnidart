"""تست خودِ ابزار پاریتی.

ابزار مقایسه، چیزی است که قضاوت «آیا ربات درست کار می‌کند» به آن گره
خورده. اگر خودِ ابزار خراب باشد، یا همه‌چیز را سبز نشان می‌دهد یا همه‌چیز
را قرمز. پس باید خودش تست شود.

این تست‌ها با لاگ‌های مصنوعی ساخته‌شده از مرجعِ واقعی کار می‌کنند:
  ۱) لاگ سالم      ⇒ باید ۱۰۰٪ یکی باشد
  ۲) لاگ با جهتِ غلط ⇒ باید دقیقاً همان تعداد را بگیرد
  ۳) لاگ با سیگنالِ حذف‌شده ⇒ باید «پیدا نشد» بدهد
  ۴) لاگ خالی       ⇒ باید پیام راهنما بدهد، نه کرش
  ۵) لاگ نسخهٔ قدیمی (بدون BUY/SELL) ⇒ باید جهت را از قیمت استنباط کند
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compare_parity as cp

LINE = ("{ts}  Esprakt_M1_LiquiditySweep (US30,M1)  📝 [شبیه‌سازی] وسط‌کندل | "
        "{side} | ورود {entry:.5f} | استاپ {stop:.5f} | هدف {tgt:.5f} | R=6.26 | حجم 0.01\n")
LINE_NO_SIDE = ("{ts}  Esprakt_M1_LiquiditySweep (US30,M1)  📝 [شبیه‌سازی] وسط‌کندل | "
                "ورود {entry:.5f} | استاپ {stop:.5f} | هدف {tgt:.5f} | R=6.26 | حجم 0.01\n")


@pytest.fixture
def ref():
    """مرجع مصنوعیِ کوچک و کنترل‌شده (وابسته به داده‌های واقعی نیست)."""
    rows = []
    t0 = pd.Timestamp("2024-03-01 08:00")
    for i in range(12):
        d = 1 if i % 2 == 0 else -1
        entry = 34_000 + i
        # استاپ همیشه در سمتِ ضرر است: خرید → زیرِ ورود، فروش → بالای ورود
        stop = entry - 10.0 if d == 1 else entry + 10.0
        rows.append({
            "time": t0 + pd.Timedelta(hours=i),
            "dir": d,
            "entry": entry,
            "entry_signal": entry - 0.04,   # قیمت لیمیت
            "stop": stop,
            "target": entry + 20.0 * d,
        })
    return pd.DataFrame(rows)


def write_log(path, df, side=True, flip=(), drop=()):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in df.iterrows():
            if i in drop:
                continue
            d = -r["dir"] if i in flip else r["dir"]
            tpl = LINE if side else LINE_NO_SIDE
            f.write(tpl.format(ts=r["time"].strftime("%Y.%m.%d %H:%M:%S"),
                               side="BUY" if d == 1 else "SELL",
                               entry=r["entry_signal"], stop=r["stop"], tgt=r["target"]))
    return path


def run(ref, log):
    ref = ref.copy()
    ref["side"] = ref["dir"].map(lambda v: "BUY" if v == 1 else "SELL")
    got = cp.parse_log(log)
    assert not got.empty, "لاگ خوانده نشد"
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cp.match(ref, got, tol_min=5.0, tol_price=10.0)
    return buf.getvalue()


def test_clean_log_is_full_parity(ref, tmp_path):
    out = run(ref, write_log(tmp_path / "ok.log", ref))
    assert "✅ یکی: 12/12" in out
    assert "🟢" in out


def test_wrong_direction_is_caught(ref, tmp_path):
    flip = {0, 3, 7, 11}
    out = run(ref, write_log(tmp_path / "flip.log", ref, flip=flip))
    assert "جهت غلط: 4" in out
    assert "✅ یکی: 8/12" in out


def test_missing_signals_are_reported(ref, tmp_path):
    drop = {2, 5}
    out = run(ref, write_log(tmp_path / "drop.log", ref, drop=drop))
    assert "پیدا نشد: 2" in out


def test_old_format_derives_direction(ref, tmp_path):
    """نسخهٔ قدیمی ربات BUY/SELL نمی‌نوشت؛ ابزار باید از قیمت بفهمد."""
    out = run(ref, write_log(tmp_path / "old.log", ref, side=False))
    assert "✅ یکی: 12/12" in out


def test_empty_log_raises_helpful_error(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("", encoding="utf-8")
    got = cp.parse_log(str(p))
    assert got.empty


def test_parser_ignores_non_signal_lines(tmp_path):
    p = tmp_path / "noise.log"
    p.write_text(
        "2024.03.01 08:00:00   Esprakt  ── ربات اسپرکت راه‌اندازی شد ──\n"
        "2024.03.01 08:00:01   ATR handle failed\n"
        "2024.03.01 08:00:02   ⏱ خروج زمانی پس از 60 کندل (RR نرسید)\n",
        encoding="utf-8")
    assert cp.parse_log(str(p)).empty


def test_gap_through_stop_does_not_break_direction(ref, tmp_path):
    """وقتی قیمت از استاپ پر شود، entry پرشده از استاپ می‌گذرد ولی
    جهتِ سیگنال (entry_signal) همچنان درست و قابل تشخیص است."""
    d = ref.copy()
    # قیمت گپ می‌کند و زیرِ استاپِ خرید پر می‌شود ⇒ entry پرشده < stop
    d.loc[0, "entry"] = d.loc[0, "stop"] - 5.0
    p = tmp_path / "gap.log"
    with open(p, "w", encoding="utf-8") as f:
        for i, r in d.iterrows():
            price = r["entry"] if i != 0 else d.loc[0, "entry"]
            f.write(LINE.format(ts=r["time"].strftime("%Y.%m.%d %H:%M:%S"),
                                side="BUY" if r["dir"] == 1 else "SELL",
                                entry=r["entry_signal"], stop=r["stop"], tgt=r["target"]))
    out = run(d, str(p))
    # سطرِ گپ‌خورده نباید «جهت غلط» بدهد، چون جهت از entry_signal می‌آید
    assert "جهت غلط: 0" in out
