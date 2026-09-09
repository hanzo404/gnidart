import sqlite3
import pandas as pd
import json
import time

DB_NAME = "bot_trades.db"
RULES_FILE = "dynamic_rules.json"

def get_recent_trades(limit=10):
    """دریافت آخرین معاملات انجام‌شده از دیتابیس"""
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM trades ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_losses_and_adapt(df):
    """تحلیل ریشه‌ای استاپ‌ها و به‌روزرسانی قوانین سیستم"""
    if df.empty:
        return
    
    # بررسی معاملات اخیر با نتیجه منفی
    consecutive_losses = 0
    for _, row in df.iterrows():
        if row['profit'] is not None and row['profit'] < 0:
            consecutive_losses += 1
        else:
            break

    print(f"📊 [Post-Mortem] تعداد استاپ‌های متوالی اخیر: {consecutive_losses}")

    # تعیین قوانین جدید بر اساس تحلیل خطا (Self-Correction Logic)
    rules = {
        "strict_mode": False,
        "cooldown_multiplier": 1,
        "min_rr_ratio": 1.5,
        "reason": "عملکرد طبیعی سیستم"
    }

    if consecutive_losses >= 3:
        rules["strict_mode"] = True
        rules["cooldown_multiplier"] = 2  # دو برابر کردن زمان انتظار بین معاملات
        rules["min_rr_ratio"] = 2.0       # افزایش حداقل ریسک به ریوارد برای ورودهای با اطمینان بالا
        rules["reason"] = f"شناسایی {consecutive_losses} استاپ متوالی. فعال‌سازی مد سخت‌گیرانه جهت جلوگیری از Overtrading."
        print(f"⚠️ [Self-Correction Alert] {rules['reason']}")

    # ذخیره‌سازی قوانین پویای جدید جهت استفاده توسط موتور اجرای سریع
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=4)

def run_post_mortem_loop():
    """حلقه پایش و کالبدشکافی دوره ای در پس‌زمینه"""
    print("🔬 ماژول Post-Mortem Analyst (کالبدشکافی و یادگیری خطا) فعال شد...")
    while True:
        try:
            df_trades = get_recent_trades(limit=10)
            analyze_losses_and_adapt(df_trades)
        except Exception as e:
            print(f"❌ خطا در تحلیل Post-Mortem: {e}")
        
        # پایش و تحلیل هر ۵ دقیقه یک بار (بدون ایجاد تاخیر روی موتور اسکلپ)
        time.sleep(300)

if __name__ == "__main__":
    run_post_mortem_loop()