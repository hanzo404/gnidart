import MetaTrader5 as mt5
import pandas as pd
import time
import json
import os

SYMBOL = "XAUUSD"
BIAS_FILE = "market_bias.json"

def get_high_tf_data(symbol, timeframe, num_candles=50):
    """دریافت داده‌های تایم‌فریم بالا (H4) جهت تعیین ساختار کلان"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def analyze_market_structure(df):
    """شناسایی BOS و CHoCH جهت تعیین بایاس اصلی بازار (SMC H4)"""
    if len(df) < 10:
        return "NEUTRAL"
    
    # استخراج سقف‌ها و کف‌های اصلی ۱۰ کندل اخیر
    recent_highs = df['high'].tail(10)
    recent_lows = df['low'].tail(10)
    
    current_close = df['close'].iloc[-1]
    prev_high_max = recent_highs.iloc[:-1].max()
    prev_low_min = recent_lows.iloc[:-1].min()

    # شکست سقف (BOS صعودی)
    if current_close > prev_high_max:
        return "BULLISH"
    # شکست کف (BOS نزولی)
    elif current_close < prev_low_min:
        return "BEARISH"
    
    return "NEUTRAL"

def update_bias_file(bias_direction):
    """ذخیره وضعیت تحلیل در یک فایل سبک جهت اشتراک‌گذاری با موتور اجرایی"""
    bias_data = {
        "symbol": SYMBOL,
        "bias": bias_direction,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(BIAS_FILE, "w") as f:
        json.dump(bias_data, f, indent=4)
    print(f"🧠 [مغز هوشمند AI] وضعیت جدید بایاس بازار ثبت شد: {bias_direction}")

def run_ai_brain():
    """حلقه اصلی تحلیل تایم‌فریم بالا بدون درگیر کردن سرعت اسکلپ"""
    print("🧠 ماژول تحلیل‌گر AI Brain روی تایم‌فریم H4 فعال شد...")
    
    if not mt5.initialize():
        print("❌ عدم توانایی در اتصال به MT5")
        return

    while True:
        # دریافت کندل‌های H4
        df_h4 = get_high_tf_data(SYMBOL, mt5.TIMEFRAME_H4, 50)
        
        if df_h4 is not None:
            # تحلیل ساختار کلان بازار
            bias = analyze_market_structure(df_h4)
            update_bias_file(bias)
        
        # بروزرسانی جهت هر ۳۰ دقیقه یک‌بار (بدون اشغال حافظه و تاخیر)
        time.sleep(1800)

if __name__ == "__main__":
    run_ai_brain()