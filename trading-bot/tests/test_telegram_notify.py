"""تست‌های فاز ops — اطلاع‌رسانی تلگرام (بدون شبکه: urllib ماک می‌شود)."""
import json
import unittest
from unittest import mock

from bot.live.telegram_notify import MAX_LEN, TelegramNotifier


def _ok_response():
    m = mock.MagicMock()
    m.__enter__.return_value.status = 200
    return m


class TestTelegramNotifier(unittest.TestCase):
    def test_disabled_is_noop(self):
        """نبود توکن/چت = هیچ تماس شبکه‌ای نباید صورت بگیرد."""
        n = TelegramNotifier()
        self.assertFalse(n.enabled)
        with mock.patch("urllib.request.urlopen") as u:
            self.assertFalse(n.send("سلام"))
            u.assert_not_called()

    def test_send_posts_json(self):
        n = TelegramNotifier("123:ABC", 42)
        self.assertTrue(n.enabled)
        with mock.patch("urllib.request.urlopen",
                        return_value=_ok_response()) as u:
            self.assertTrue(n.send("سلام بات"))
            req = u.call_args[0][0]
            self.assertIn("123:ABC", req.full_url)
            self.assertIn("sendMessage", req.full_url)
            payload = json.loads(req.data.decode("utf-8"))
            self.assertEqual(payload["chat_id"], "42")
            self.assertEqual(payload["text"], "سلام بات")

    def test_network_error_swallowed(self):
        """شکست شبکه هرگز نباید raise کند (حلقهٔ معاملاتی مقدس است)."""
        n = TelegramNotifier("123:ABC", 42)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            self.assertFalse(n.send("اول"))   # هشدار چاپ می‌شود
            self.assertFalse(n.send("دوم"))   # ساکت (فقط یک هشدار)

    def test_long_text_truncated(self):
        n = TelegramNotifier("123:ABC", 42)
        with mock.patch("urllib.request.urlopen",
                        return_value=_ok_response()) as u:
            n.send("x" * 10000)
            payload = json.loads(u.call_args[0][0].data.decode("utf-8"))
            self.assertEqual(len(payload["text"]), MAX_LEN)

    def test_from_config_missing_file(self):
        n = TelegramNotifier.from_config(path="/nonexistent/x.yaml")
        self.assertFalse(n.enabled)

    def test_from_config_reads_yaml(self):
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "t.yaml"
            p.write_text("bot_token: '9:Z'\nchat_id: 77\n", encoding="utf-8")
            n = TelegramNotifier.from_config(path=p)
            self.assertTrue(n.enabled)
            self.assertEqual(n.chat_id, "77")


if __name__ == "__main__":
    unittest.main()
