import threading
import time
import MetaTrader5 as mt5
from ai_brain import run_ai_brain
from post_mortem_analyst import run_post_mortem_loop
from fast_execution import get_market_data, check_m1_trigger, calculate_dynamic_sl_tp, execute_scalp_trade, SYMBOL, TIMEFRAME
from trade_logger import init_db

def run_fast_execution_loop():
    """حلقه اصلی اجرای سریع اسکلپ روی M1"""
    print("⚡ موتور اسکلپ سریع M1 فعال شد...")
    while True:
        try:
            df_m1 = get_market_data(SYMBOL, TIMEFRAME, 50)
            if df_m1 is not None:
                action = check_m1_trigger(df_m1)
                if action:
                    price = mt5.symbol_info_tick(SYMBOL).ask if action == 'BUY' else mt5.symbol_info_tick(SYMBOL).bid
                    sl, tp = calculate_dynamic_sl_tp(df_m1, action, price)
                    execute_scalp_trade(action, SYMBOL, price, sl, tp)
        except Exception as e:
            print(f"❌ خطا در اجرای سریع: {e}")
        
        time.sleep(3)  # چک کردن چارت M1 هر ۳ ثانیه یک‌بار

if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ اتصال به MT5 برقرار نشد!")
        exit()
        
    init_db()
    print("🚀 سیستم معامله‌گر عامل‌محور (Agentic AI System) در حال راه‌اندازی است...\n")

    # اجرای ماژول‌ها در ترد‌های (Thread) مجزا برای جلوگیری از دیلی
    t_brain = threading.Thread(target=run_ai_brain, daemon=True)
    t_post_mortem = threading.Thread(target=run_post_mortem_loop, daemon=True)
    
    t_brain.start()
    t_post_mortem.start()

    # اجرای موتور اصلی در ترد اصلی
    run_fast_execution_loop()