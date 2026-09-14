"""اطلاع‌رسانی تلگرام — فاز ops (۲۰۲۶-۰۹-۱۵).

معماری: این ماژول هیچ چیزی دربارهٔ استراتژی نمی‌داند — فقط «متن بفرست».
فرستادنِ پیام هرگز حلقهٔ معاملاتی را نمی‌کشد: هر خطا بلعیده می‌شود و
فقط یک‌بار در عمرِ فرایند روی کنسول هشدار می‌گیرد (همان فلسفهٔ
اعتبارسنجی مشخصات نماد — قابلیت جانبی نباید هسته را بکشد).

فقط stdlib (urllib) — هیچ وابستگی جدیدی روی VPS لازم نیست.

راه‌اندازی: py scripts/telegram_setup.py  →  config/telegram.local.yaml
نبودِ آن فایل = اطلاع‌رسانِ خاموش (سازگار کامل با بکاپ خونه و تست‌ها).
"""
from __future__ import annotations

import json
import pathlib
import ssl
import urllib.request

try:  # هات‌فیکس SSL ویندوز (۲۰۲۶-۰۹-۱۵): مخزن گواهیِ پایتون روی
    import certifi  # سرور ۲۰۲۲ می‌تواند ریشهٔ تلگرام را نداشته باشد؛
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())  # curl
except ImportError:  # به همین دلیل کار می‌کرد. certifi = همان باندل Mozilla.
    SSL_CTX = None  # نبود certifi → رفتار قبلی (مخزن سیستم)

# سقف پیام تلگرام ۴۰۹۶ کاراکتر است؛ حاشیهٔ اطمینان می‌گذاریم
MAX_LEN = 4000
_TIMEOUT = 10
CONFIG_PATH = pathlib.Path("config/telegram.local.yaml")


class TelegramNotifier:
    """پیام‌رسان سبک؛ بدون تنظیمات = no-op بی‌صدا."""

    def __init__(self, token: str | None = None,
                 chat_id: str | int | None = None):
        self.token = (token or "").strip()
        self.chat_id = (str(chat_id).strip() if chat_id is not None else "")
        self.enabled = bool(self.token and self.chat_id)
        self._warned = False  # هشدار شکست شبکه فقط یک‌بار چاپ می‌شود

    @classmethod
    def from_config(cls, path=CONFIG_PATH) -> "TelegramNotifier":
        """بارگذاری اختیاری؛ هر خطا → اطلاع‌رسان خاموش."""
        try:
            import yaml

            p = pathlib.Path(path)
            if not p.exists():
                return cls()
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cls(cfg.get("bot_token"), cfg.get("chat_id"))
        except Exception:
            return cls()

    def send(self, text: str) -> bool:
        """ارسال non-blocking-ish؛ خروجی False یعنی ارسال نشد (بدون raise)."""
        if not self.enabled:
            return False
        try:
            data = json.dumps(
                {"chat_id": self.chat_id, "text": str(text)[:MAX_LEN]}
            ).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT,
                                        context=SSL_CTX) as resp:
                return resp.status == 200
        except Exception as e:  # noqa: BLE001 — پیام‌رسان هرگز ربات را نکشد
            if not self._warned:
                print(f"(تلگرام: ارسال ناموفق — {e})")
                self._warned = True
            return False
