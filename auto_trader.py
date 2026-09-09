import time
import MetaTrader5 as mt5
from signal_generator import generate_signals

def send_order(action, symbol, entry, sl, tp, volume=0.01):
    """ارسال مستقیم دستور معامله به متاتریدر ۵"""
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
        "deviation": 20,
        "magic": 100200,
        "comment": "SMC Bot Auto-Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ خطا در ثبت معامله: {result.comment} (کد: {result.retcode})")
        return False
    else:
        print(f"✅ معامله {action} با موفقیت ثبت شد! شناسه معامله: {result.order}")
        return True

def run_bot(symbol="XAUUSD", check_interval=10):
    """چرخه پایش مداوم بازار و اجرای خودکار"""
    if not mt5.initialize():
        print("❌ عدم توانایی در اتصال به متاتریدر ۵")
        return

    print(f"🤖 ربات معامله‌گر SMC فعال شد. در حال مانیتورینگ {symbol}...")
    
    try:
        while True:
            # 1. بررسی پوزیشن‌های باز جهت جلوگیری از معامله تکراری
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) > 0:
                print("⏳ یک پوزیشن باز روی این نماد وجود دارد. ربات منتظر بسته شدن آن می‌ماند...", end="\r")
            else:
                # 2. بررسی سیگنال SMC
                signal, current_price = generate_signals(symbol)
                
                if signal:
                    print(f"\n🚀 سیگنال جدید یافت شد: {signal['action']} روی قیمت {current_price}")
                    send_order(
                        action=signal['action'],
                        symbol=symbol,
                        entry=signal['entry'],
                        sl=signal['sl'],
                        tp=signal['tp'],
                        volume=0.01  # حجم معامله
                    )
                else:
                    print(f"👁️ در حال مانیتورینگ... قیمت فعلی: {current_price} (هیچ تاچ زونی رخ نداده)", end="\r")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n🛑 ربات با دستور کاربر متوقف شد.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_bot("XAUUSD", check_interval=10)