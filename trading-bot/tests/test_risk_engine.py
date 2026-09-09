"""تست موتور ریسک: سایز پوزیشن، حد روزانه، حد ماهانه، همکاری با برق‌گیر."""
import sys
import pathlib
import unittest
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.risk.circuit_breaker import CircuitBreaker
from bot.risk.engine import RiskEngine

T0 = datetime(2026, 9, 9, 12, 0)


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.b = CircuitBreaker()
        self.eng = RiskEngine(self.b, risk_per_trade=0.005)

    def test_basic_size(self):
        # equity=10000, risk=0.5%, entry=1.1000, stop=1.0980 → فاصله 0.0020
        d = self.eng.size_for(10_000.0, 1.1000, 1.0980, "A", T0)
        self.assertTrue(d.approved)
        self.assertAlmostEqual(d.risk_amount, 50.0)
        self.assertAlmostEqual(d.units, 50.0 / 0.0020)  # 25,000 واحد

    def test_breaker_halves_size(self):
        self.b.on_trade_closed(-1, T0)
        self.b.on_trade_closed(-1, T0)
        d = self.eng.size_for(10_000.0, 1.1000, 1.0980, "A", T0)
        self.assertTrue(d.approved)
        self.assertAlmostEqual(d.units, 12_500.0)  # 25,000 × 0.5

    def test_rejects_when_deep_derate_and_grade_B(self):
        for _ in range(3):
            self.b.on_trade_closed(-1, T0)
        d = self.eng.size_for(10_000.0, 1.1000, 1.0980, "B", T0)
        self.assertFalse(d.approved)

    def test_rejects_zero_stop_distance(self):
        d = self.eng.size_for(10_000.0, 1.1000, 1.1000, "A", T0)
        self.assertFalse(d.approved)


class TestGuards(unittest.TestCase):
    def test_daily_loss_guard_blocks(self):
        b = CircuitBreaker()
        eng = RiskEngine(b, risk_per_trade=0.005, max_daily_loss=0.03)
        eng.on_equity_update(10_000.0, T0)
        eng.on_equity_update(9_600.0, T0)  # -4% در همان روز
        d = eng.size_for(9_600.0, 1.1000, 1.0980, "A", T0)
        self.assertFalse(d.approved)
        self.assertIn("daily", d.reason)

    def test_monthly_loss_halts_via_breaker(self):
        b = CircuitBreaker()
        eng = RiskEngine(b, risk_per_trade=0.005, max_monthly_loss=0.06)
        eng.on_equity_update(10_000.0, T0)
        eng.on_equity_update(9_300.0, T0)  # -7% در همان ماه
        self.assertTrue(b.halted)
        d = eng.size_for(9_300.0, 1.1000, 1.0980, "A", T0)
        self.assertFalse(d.approved)
        self.assertIn("monthly", b.state()["halt_reason"] or "")


class TestJournalIntegration(unittest.TestCase):
    def test_trade_lifecycle_and_stats(self):
        import tempfile
        import os
        from bot.journal import Journal

        with tempfile.TemporaryDirectory() as td:
            j = Journal(os.path.join(td, "t.db"))
            tid = j.open_trade(
                opened_at=T0, symbol="EURUSD", direction=+1, strategy="trend_pullback",
                grade="A", entry=1.1000, stop=1.0980, target=1.1040, size_units=25_000,
                regime="trend_weak", features={"adx": 22, "atr_pct": 0.4}, reason="test",
            )
            j.close_trade(tid, T0.replace(minute=30), 1.1040, r_multiple=+2.0,
                          mfe_r=2.1, mae_r=-0.4)
            st = j.stats()
            self.assertEqual(st["trades"], 1)
            self.assertEqual(st["win_rate"], 1.0)
            self.assertAlmostEqual(st["total_r"], 2.0)
            trades = j.recent_trades(5)
            self.assertEqual(trades[0]["features"]["adx"], 22)
            j.close()


if __name__ == "__main__":
    unittest.main()
