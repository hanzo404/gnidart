"""تست رگرسیون گزارش دمو — باگ #1: ستون‌های بی‌نام از sqlite3.Row.

صحنه‌ی باگ (واقعی، ۲۰۲۶-۰۹-۱۰): ربات یک شبانه‌روز درای‌ران کرده،
صفر معامله (رژیم نزولی + فقط-خرید)، فقط ردیف equity ثبت شده →
گزارش روی KeyError: 'equity' می‌مرد. این تست‌ها همان صحنه‌ها را
قفل می‌کنند: خالی، فقط-equity، و پر (معامله + بریکر + equity).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.journal.store import Journal

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "live_report.py"


def run_report(db_path: pathlib.Path) -> str:
    spec = importlib.util.spec_from_file_location("live_report", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv_backup = sys.argv
    sys.argv = ["live_report.py", "--db", str(db_path)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = argv_backup
    return buf.getvalue()


class TestLiveReport(unittest.TestCase):
    def test_empty_journal_no_crash(self):
        """فایل db هست ولی هیچ جدولی/داده‌ای نیست → پیام تمیز، نه کرش."""
        with tempfile.TemporaryDirectory() as td:
            db = pathlib.Path(td) / "journal.db"
            sqlite3.connect(db).close()      # فایل خالی
            out = run_report(db)
            self.assertIn("هنوز داده‌ای ثبت نشده", out)

    def test_equity_only_no_trades_no_crash(self):
        """صحنه‌ی باگ #1: فقط equity، صفر معامله → نباید KeyError بدهد."""
        with tempfile.TemporaryDirectory() as td:
            db = pathlib.Path(td) / "journal.db"
            js = Journal(str(db))
            t0 = datetime.now() - timedelta(minutes=10)
            js.record_equity(t0, 3000.0)
            js.record_equity(t0 + timedelta(minutes=5), 3001.5)
            js.close()
            out = run_report(db)
            self.assertIn("معاملات بسته‌شده: 0", out)
            self.assertIn("سرمایهٔ هر آستین", out)
            self.assertIn("XAUUSD: آخرین", out)   # ردیف بدون symbol = طلا
            self.assertIn("ضربان", out)
            self.assertIn("✅", out)              # چرخهٔ تازه → زنده

    def test_stale_heartbeat_warns(self):
        """آخرین چرخه قدیمی → هشدار خاموشی (⚠️ نه ✅)."""
        with tempfile.TemporaryDirectory() as td:
            db = pathlib.Path(td) / "journal.db"
            js = Journal(str(db))
            t0 = datetime.now() - timedelta(hours=6)
            js.record_equity(t0, 3000.0)
            js.record_equity(t0 + timedelta(minutes=15), 2999.0)
            js.close()
            out = run_report(db)
            self.assertIn("ضربان", out)
            self.assertIn("⚠️", out)
            self.assertNotIn("✅", out)

    def test_full_journal_renders(self):
        """معامله + بریکر + equity → همه بخش‌ها بدون خطا."""
        with tempfile.TemporaryDirectory() as td:
            db = pathlib.Path(td) / "journal.db"
            js = Journal(str(db))
            t0 = datetime(2026, 9, 10, 15, 0)
            tid = js.open_trade(t0, "XAUUSD", 1, "fvg", "A",
                                4400.0, 4390.0, 4425.0, 0.01,
                                regime="TREND_UP")
            js.close_trade(tid, t0 + timedelta(hours=2), 4425.0, 2.1)
            js.record_breaker_event(t0 + timedelta(hours=3), "derate",
                                    "رشتهٔ ۳ باخت — تست",
                                    streak=3, size_multiplier=0.5)
            js.record_equity(t0, 3000.0)
            js.record_equity(t0 + timedelta(hours=3), 3060.0)
            js.close()
            out = run_report(db)
            self.assertIn("معاملات بسته‌شده: 1", out)
            self.assertIn("آستین XAUUSD", out)
            self.assertIn("نرخ برد", out)
            self.assertIn("رویدادهای بریکر", out)
            self.assertIn("XAUUSD: آخرین $", out)

    def test_multi_symbol_sections(self):
        """فاز ۷: طلا + نقره در یک ژورنال → هر آستین بخش و ضربان خودش."""
        with tempfile.TemporaryDirectory() as td:
            db = pathlib.Path(td) / "journal.db"
            js = Journal(str(db))
            t0 = datetime.now() - timedelta(minutes=10)
            tid = js.open_trade(t0, "XAGUSD", 1, "fvg", "A",
                                74.10, 74.00, 74.35, 0.01,
                                regime="TREND_UP")
            js.close_trade(tid, t0 + timedelta(hours=1), 74.35, 2.5)
            js.record_equity(t0, 3000.0, symbol="XAUUSD")
            js.record_equity(t0 + timedelta(minutes=5), 3000.0,
                             symbol="XAGUSD")
            js.close()
            out = run_report(db)
            self.assertIn("آستین XAGUSD", out)          # بخش جدا برای نقره
            self.assertIn("XAUUSD: آخرین چرخه", out)    # ضربان جدا برای طلا
            self.assertIn("XAGUSD: آخرین چرخه", out)    # ضربان جدا برای نقره
            self.assertIn("سرمایهٔ هر آستین", out)
            self.assertIn("معاملات بسته‌شده: 1", out)


if __name__ == "__main__":
    unittest.main()
