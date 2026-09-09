import MetaTrader5 as mt5
import pandas as pd
from smc_logic import fetch_market_data, find_fair_value_gaps, find_order_blocks

def generate_signals(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, risk_reward_ratio=3.0):
    """
    بررسی برخورد قیمت جاری به نواحی SMC و صدور سیگنال با R:R مشخص
    """
    df = fetch_market_data(symbol, timeframe, 200)
    if df is None or df.empty:
        return None

    current_price = df['close'].iloc[-1]
    fvgs = find_fair_value_gaps(df)
    obs = find_order_blocks(df)

    signal = None

    # 1. بررسی سیگنال خرید (Bullish Setup)
    # برخورد قیمت به کف FVG صعودی یا Order Block صعودی
    latest_bullish_obs = obs[obs['type'] == 'BULLISH_OB'] if not obs.empty else pd.DataFrame()
    if not latest_bullish_obs.empty:
        last_ob = latest_bullish_obs.iloc[-1]
        # اگر قیمت در محدوده اردر بلاک صعودی باشد
        if last_ob['bottom'] <= current_price <= last_ob['top']:
            sl = last_ob['bottom'] - 1.0  # حد ضرر کمی پایین‌تر از کف OB
            risk = current_price - sl
            tp = current_price + (risk * risk_reward_ratio) # حد سود با R:R تعیین شده
            
            signal = {
                'action': 'BUY',
                'symbol': symbol,
                'entry': current_price,
                'sl': round(sl, 2),
                'tp': round(tp, 2),
                'reason': 'برخورد قیمت به Order Block صعودی'
            }

    # 2. بررسی سیگنال فروش (Bearish Setup)
    # برخورد قیمت به محدوده Order Block نزولی
    latest_bearish_obs = obs[obs['type'] == 'BEARISH_OB'] if not obs.empty else pd.DataFrame()
    if not latest_bearish_obs.empty and signal is None:
        last_ob = latest_bearish_obs.iloc[-1]
        # اگر قیمت در محدوده اردر بلاک نزولی باشد
        if last_ob['bottom'] <= current_price <= last_ob['top']:
            sl = last_ob['top'] + 1.0  # حد ضرر کمی بالاتر از سقف OB
            risk = sl - current_price
            tp = current_price - (risk * risk_reward_ratio)
            
            signal = {
                'action': 'SELL',
                'symbol': symbol,
                'entry': current_price,
                'sl': round(sl, 2),
                'tp': round(tp, 2),
                'reason': 'برخورد قیمت به Order Block نزولی'
            }

    return signal, current_price

if __name__ == "__main__":
    if mt5.initialize():
        symbol = "XAUUSD"
        print(f"--- بررسی زنده سیگنال‌های SMC برای {symbol} ---")
        
        sig, price = generate_signals(symbol)
        print(f"قیمت جاری طلا: {price}")
        
        if sig:
            print("\n🎯 **سیگنال جدید صادر شد!**")
            print(f"نوع معامله: {sig['action']}")
            print(f"نقطه ورود: {sig['entry']}")
            print(f"حد ضرر (SL): {sig['sl']}")
            print(f"حد سود (TP): {sig['tp']}")
            print(f"دلیل ورود: {sig['reason']}")
        else:
            print("\n⏳ هیچ سیگنال فعالی در این لحظه یافت نشد (قیمت در نواحی حساس SMC نیست).")
            
        mt5.shutdown()