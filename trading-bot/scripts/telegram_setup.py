"""راه‌اندازی اطلاع‌رسانی تلگرام — تعاملی و گام‌به‌گام (فاز ops، ۲۰۲۶-۰۹-۱۵).

اجرا (روی ویندوز، از ریشهٔ پروژه):
    py scripts/telegram_setup.py

چهار قدم:
  ۱) با @BotFather بات بساز و توکنش را کپی کن (توکن = راز؛ در چت نبر!)
  ۲) بات را در تلگرام خودت Start کن (بدون این، بات نمی‌تواند به تو پیام دهد)
  ۳) این اسکریپت chat_id را خودش پیدا می‌کند و پیام آزمایشی می‌فرستد
  ۴) config/telegram.local.yaml نوشته می‌شود (هرگز به گیت نمی‌رود)
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.live.telegram_notify import (  # noqa: E402
    CONFIG_PATH, SSL_CTX, TelegramNotifier,
)


def api(token: str, method: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    # SSL_CTX = باندل certifi (Mozilla) — روی Server 2022 مخزن سیستم
    # می‌تواند ناقص باشد و getUpdates با CERTIFICATE_VERIFY_FAILED بمیرد.
    with urllib.request.urlopen(url, timeout=10, context=SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print("=" * 56)
    print("راه‌اندازی اطلاع‌رسانی تلگرام برای gnidart")
    print("=" * 56)

    print("\nقدم ۱/۴ — اگر هنوز بات نساخته‌ای:")
    print("  • توی تلگرام به @BotFather پیام بده: /newbot")
    print("  • یک اسم نمایشی بده (مثلاً: gnidart notifier)")
    print("  • یک username بده که به bot ختم شود (مثلاً: my_gnidart_bot)")
    print("  • توکن را کپی کن — شبیه: 123456789:AAHx...")
    print("  ⚠️ توکن را با هیچ‌کس (حتی دستیار هوش مصنوعی!) به اشتراک نگذار.")
    token = input("\nتوکن را اینجا پیست کن و Enter بزن: ").strip()
    if ":" not in token:
        sys.exit("❌ این شبیه توکن نیست — باید شکل «عدد:حروف» باشد. دوباره اجرا کن.")

    print("\nقدم ۲/۴ — بات جدیدت را در تلگرام باز کن و دکمهٔ START را بزن")
    print("  (یا /start بفرست). بدون این مرحله بات اجازهٔ پیام دادن به تو را ندارد.")
    input("زدم! Enter بزن... ")

    print("\nقدم ۳/۴ — دنبال chat_id می‌گردم...")
    chat_id = None
    while chat_id is None:
        try:
            res = api(token, "getUpdates")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"❌ getUpdates ناموفق ({e}) — توکن را چک کن و دوباره اجرا کن.")
        for upd in reversed(res.get("result") or []):
            msg = upd.get("message") or upd.get("channel_post")
            if msg and msg.get("chat", {}).get("id"):
                chat_id = msg["chat"]["id"]
                break
        if chat_id is None:
            input("چیزی پیدا نشد — مطمئن شو به بات پیام داده‌ای، بعد Enter: ")

    print(f"  ✅ chat_id پیدا شد: {chat_id}")
    n = TelegramNotifier(token, chat_id)
    if not n.send("✅ gnidart به تلگرام وصل شد! این یک پیام آزمایشی است."):
        sys.exit("❌ پیام آزمایشی نرسید — اینترنت VPS/توکن را چک کن و دوباره اجرا کن.")
    print("  ✅ پیام آزمایشی رفت — تلگرامت را چک کن!")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# ایجادشده توسط scripts/telegram_setup.py\n"
        "# توکن راز است — این فایل هرگز کامیت نمی‌شود (.gitignore)\n"
        f"bot_token: {token}\n"
        f"chat_id: {chat_id}\n",
        encoding="utf-8",
    )
    print(f"\nقدم ۴/۴ ✅ ذخیره شد: {CONFIG_PATH}")
    print("\nآخرین قدم: رانرها را ری‌استارت کن — در هر پنجره Ctrl+C،")
    print("بعد دوباره start.bat — و پیام «شروع شد» به تلگرامت می‌آید 🚀")


if __name__ == "__main__":
    main()
