"""تست‌های نردبان برق‌گیر ضرر — سناریوهای واقعی رشته باخت/برد."""
import sys
import pathlib
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.risk.circuit_breaker import CircuitBreaker

T0 = datetime(2026, 9, 9, 12, 0)
H = timedelta(hours=1)


class TestLadder(unittest.TestCase):
    def test_two_losses_derate_half(self):
        b = CircuitBreaker()
        b.on_trade_closed(-1, T0)
        b.on_trade_closed(-1, T0 + H)
        self.assertEqual(b.size_multiplier(), 0.5)
        ok, _ = b.allow_entry("B", T0 + 2 * H)
        self.assertTrue(ok, "در سطح ۲ هنوز ستاپ B مجاز است")

    def test_three_losses_deep_derate_a_only(self):
        b = CircuitBreaker()
        for i in range(3):
            b.on_trade_closed(-1, T0 + i * H)
        self.assertEqual(b.size_multiplier(), 0.25)
        self.assertFalse(b.allow_entry("B", T0 + 4 * H)[0])
        self.assertTrue(b.allow_entry("A", T0 + 4 * H)[0])

    def test_four_losses_pause_24h(self):
        b = CircuitBreaker()
        for i in range(4):
            b.on_trade_closed(-1, T0 + i * H)
        ok, why = b.allow_entry("A", T0 + 5 * H)
        self.assertFalse(ok)
        self.assertIn("paused", why)
        # پایان مهلت: توقف از باخت چهارم (T0+3h) شروع شد + ۲۴h = T0+27h
        ok2, _ = b.allow_entry("A", T0 + 28 * H)
        self.assertTrue(ok2)
        self.assertEqual(b.size_multiplier(), 0.25)

    def test_six_losses_full_halt_needs_ack(self):
        b = CircuitBreaker()
        for i in range(6):
            b.on_trade_closed(-1, T0 + i * H)
        self.assertTrue(b.halted)
        ok, _ = b.allow_entry("A", T0 + 50 * H)
        self.assertFalse(ok, "حتی بعد از مهلت توقف، halt فقط با acknowledge باز می‌شود")
        b.acknowledge(T0 + 60 * H)
        self.assertFalse(b.halted)
        # بعد از ack هم با حجم کم شروع می‌کند — نه پرش به حجم کامل
        self.assertEqual(b.size_multiplier(), 0.25)

    def test_win_restores_one_step_at_a_time(self):
        b = CircuitBreaker()
        for i in range(3):
            b.on_trade_closed(-1, T0 + i * H)          # 0.25
        b.on_trade_closed(+1.8, T0 + 3 * H)            # برد در حجم کم → 0.5
        self.assertEqual(b.size_multiplier(), 0.5)
        b.on_trade_closed(+1.5, T0 + 4 * H)            # برد دوم → 1.0
        self.assertEqual(b.size_multiplier(), 1.0)
        self.assertEqual(b.streak, 0)

    def test_breakeven_is_neutral(self):
        b = CircuitBreaker()
        b.on_trade_closed(-1, T0)
        b.on_trade_closed(-1, T0 + H)                   # streak=2, x0.5
        b.on_trade_closed(0.0, T0 + 2 * H)              # سربه‌سر: هیچ تغییری
        self.assertEqual(b.size_multiplier(), 0.5)
        self.assertEqual(b.streak, 2)
        b.on_trade_closed(-1, T0 + 3 * H)               # streak=3 → عمیق
        self.assertEqual(b.size_multiplier(), 0.25)

    def test_monthly_drawdown_halts(self):
        b = CircuitBreaker(max_monthly_dd=0.06)
        ev = b.on_monthly_drawdown(-0.07, T0)
        self.assertIsNotNone(ev)
        self.assertTrue(b.halted)
        # دراودان کمتر از حد: ساکت
        b2 = CircuitBreaker(max_monthly_dd=0.06)
        self.assertIsNone(b2.on_monthly_drawdown(-0.05, T0))
        self.assertFalse(b2.halted)

    def test_events_are_journaled(self):
        b = CircuitBreaker()
        b.on_trade_closed(-1, T0)
        b.on_trade_closed(-1, T0 + H)
        kinds = [e.kind for e in b.events]
        self.assertIn("derate", kinds)
        self.assertIn("diagnostic_queued", kinds)
        # باخت اول (سطح ۱) رویدادی ندارد؛ باخت دوم: derate + diagnostic
        self.assertEqual(len(b.events), 2)


if __name__ == "__main__":
    unittest.main()
