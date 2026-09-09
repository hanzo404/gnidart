import time
import json
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from trade_logger import init_db, log_trade, update_trade_result

# تنظیمات اصلی موتور اسکلپ
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1  # تایم‌فریم پایین برای اسکلپ سریع
COOLDOWN_SECONDS = 300        # ۵ دقیقه تاخیر اجباری بین معاملات جهت جلوگیری از اورتردینگ
last_trade_time = 0

def get_market_data(symbol, timeframe, num_candles=100):
    """دریافت کندل‌های زنده از MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_current_bias():
    """خواندن آخرین جهت بازار تعیین‌شده توسط AI Brain"""
    if os.path.exists("market_bias.json"):
        try:
            with open("market_bias.json", "r") as f:
                data = json.load(f)
                return data.get("bias", "NEUTRAL")
        except Exception:
            return "NEUTRAL"
    return "NEUTRAL"

def calculate_dynamic_sl_tp(df, action, entry_price):
    """محاسبه حد ضرر پویا بر اساس سقف/کف کندل‌های اخیر (M1)"""
    recent_candles = df.tail(5)
    if action == 'BUY':
        sl = recent_candles['low'].min() - 0.50  # ۵۰ سنت پایین‌تر از کف اخیر
        risk = entry_price - sl
        tp = entry_price + (risk * 2.5)          # ریسک به ریوارد ۱ به ۲.۵
    else: # SELL
        sl = recent_candles['high'].max() + 0.50 # ۵۰ سنت بالاتر از سقف اخیر
        risk = sl - entry_price
        tp = entry_price - (risk * 2.5)
    return round(sl, 2), round(tp, 2)

def check_m1_trigger(df):
    """تشخیص سریع CHoCH و FVG در تایم M1 با اعمال فیلتر بایاس AI و Cooldown"""
    global last_trade_time
    
    # فیلتر ۱: چک کردن شرط Cooldown برای جلوگیری از اسپم معامله
    current_time = time.time()
    if current_time - last_trade_time < COOLDOWN_SECONDS:
        return None

    if len(df) < 5:
        return None
    
    current_bias = get_current_bias()
    c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
    
    # FVG صعودی - فقط در صورتی اجازه ورود دارد که بایاس BULLISH یا NEUTRAL باشد
    if c3['low'] > c1['high'] and current_bias in ['BULLISH', 'NEUTRAL']:
        return 'BUY'
    
    # FVG نزولی - فقط در صورتی اجازه ورود دارد که بایاس BEARISH یا NEUTRAL باشد
    elif c3['high'] < c1['low'] and current_bias in ['BEARISH', 'NEUTRAL']:
        return 'SELL'
        
    return None

def execute_scalp_trade(action, symbol, entry_price, sl, tp, volume=0.01):
    """ارسال آنی معامله به متاتریدر ۵ بدون دیلی"""
    global last_trade_time
    
    order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == 'BUY' else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 990011,
        "comment": "M1 Fast Scalp AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        last_trade_time = time.time()  # به‌روزرسانی تایمر تاخیر پس از معامله موفق
        print(f"\n⚡ [موتور سریع] معامله اسکلپ {action} روی قیمت {price} صادر شد. SL: {sl} | TP: {tp}")
        db_id = log_trade(symbol, action, entry_price, sl, tp, f"M1 FVG Trigger")
        return {"order_id": result.order, "db_id": db_id}
    else:
        print(f"❌ خطا در اجرای سریع سفارش: {result.comment}")
        return None

if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ عدم توانایی در اتصال به MT5")
        exit()
    init_db()
    print("⚡ موتور اجرای سریع اسکلپ (M1 Fast Execution Engine) فعال شد...")