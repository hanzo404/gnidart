from trade_logger import get_recent_results

def evaluate_risk_rules():
    """
    بررسی ۲ معامله آخر: اگر هر دو معامله ضرر (LOSS) بوده باشند،
    سیستم محدودیت‌ها و فیلترهای سخت‌گیرانه‌تری برای معامله سوم اعمال می‌کند.
    """
    recent_trades = get_recent_results(limit=2)
    
    # اگر هنوز ۲ معامله انجام نشده باشد
    if len(recent_trades) < 2:
        return {"status": "NORMAL", "extra_confirmation": False, "volume_multiplier": 1.0}

    # بررسی اینکه آیا هر دو معامله اخیر ضرر بوده‌اند
    losses = recent_trades[recent_trades['result'] == 'LOSS']
    
    if len(losses) == 2:
        print("\n⚠️ **هشدار هوش مصنوعی:** دو معامله اخیر با استاپ مواجه شدند.")
        print("🧠 فعال‌سازی مکانیزم Self-Correction برای معامله سوم:")
        print(" - نیاز به تاییدیه‌های اضافه (تایید کندل انگالفینگ و عدم معامله صرفاً با لمس زون)")
        print(" - کاهش ۲ برابری حجم معامله جهت کنترل ریسک")
        
        return {
            "status": "STRICT_MODE",
            "extra_confirmation": True, # الزام به تاییدیه بیشتر
            "volume_multiplier": 0.5    # ریسک نصف می‌شود
        }
    
    return {"status": "NORMAL", "extra_confirmation": False, "volume_multiplier": 1.0}