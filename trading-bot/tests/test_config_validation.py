"""تست‌های اعتبارسنجی کانفیگ — v0.5.8 (ممیزی ۴، P1-5).

دو درس که این تست‌ها قفل می‌کنند:
  1) کلید ناشناخته در YAML باید «بلند» نادیده گرفته شود، نه بی‌صدا
     (غلط تایپی risk_per_tradee دیگر به تله نمی‌شود).
  2) مقدار رشته‌ای («0.005») هم‌نوع‌سازی صریح می‌شود؛ نشد → هشدار +
     پیش‌فرض می‌ماند. قبلاً بی‌صدا set می‌شد و بعداً TypeError می‌داد.
"""
import tempfile
import unittest
from pathlib import Path

from bot.config import BotConfig


def _write(tmp, text: str) -> str:
    p = Path(tmp) / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestUnknownKeys(unittest.TestCase):
    def test_unknown_section_key_warns_and_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
risk:
  risk_per_tradee: 0.01   # غلط تایپی — کلید واقعی risk_per_trade است
  risk_per_trade: 0.007
""")
            with self.assertLogs("bot.config", level="WARNING") as lg:
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.risk.risk_per_trade, 0.007)
            self.assertTrue(any("risk_per_tradee" in m for m in lg.output))

    def test_unknown_profile_key_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
symbol_profiles:
  XAUUSD:
    contract_size: 100.0
    stop_pad: 0.9        # کلید ناشناخته — sl_pad درستش است
""")
            with self.assertLogs("bot.config", level="WARNING") as lg:
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.symbol_profiles["XAUUSD"].sl_pad, 0.50)
            self.assertTrue(any("stop_pad" in m for m in lg.output))


class TestTypeCoercion(unittest.TestCase):
    def test_string_float_is_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
risk:
  risk_per_trade: "0.005"   # YAML بدون نقل‌ماننده — رشته می‌آید
""")
            cfg = BotConfig.load(path)
            self.assertIsInstance(cfg.risk.risk_per_trade, float)
            self.assertEqual(cfg.risk.risk_per_trade, 0.005)

    def test_string_int_is_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
trading:
  max_trades_per_day: "3"
""")
            cfg = BotConfig.load(path)
            self.assertIsInstance(cfg.trading.max_trades_per_day, int)
            self.assertEqual(cfg.trading.max_trades_per_day, 3)

    def test_int_given_to_float_field_is_coerced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
risk:
  max_daily_loss: 3      # int — فیلد float است
""")
            cfg = BotConfig.load(path)
            self.assertIsInstance(cfg.risk.max_daily_loss, float)
            self.assertEqual(cfg.risk.max_daily_loss, 3.0)

    def test_garbage_value_warns_and_default_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
risk:
  risk_per_trade: "صفر پینج صدم"
""")
            with self.assertLogs("bot.config", level="WARNING"):
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.risk.risk_per_trade, 0.005)  # پیش‌فرض ماند

    def test_non_integer_for_int_field_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
trading:
  max_trades_per_day: 2.5
""")
            with self.assertLogs("bot.config", level="WARNING"):
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.trading.max_trades_per_day, 3)  # پیش‌فرض

    def test_bool_for_numeric_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
trading:
  max_trades_per_day: true
""")
            with self.assertLogs("bot.config", level="WARNING"):
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.trading.max_trades_per_day, 3)

    def test_valid_config_stays_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, """
risk:
  risk_per_trade: 0.006
  max_daily_loss: 0.03
""")
            with self.assertNoLogs("bot.config", level="WARNING"):
                cfg = BotConfig.load(path)
            self.assertEqual(cfg.risk.risk_per_trade, 0.006)


if __name__ == "__main__":
    unittest.main()
