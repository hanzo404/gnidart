import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def fetch_market_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, num_bars=200):
    """دریافت داده‌های کندل‌ها از متاتریدر ۵"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        print(f"❌ خطا در دریافت داده‌های نماد {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def find_fair_value_gaps(df):
    """شناسایی شکاف‌های ارزش منصفانه (FVG)"""
    fvg_list = []
    for i in range(2, len(df)):
        # Bullish FVG
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_list.append({
                'time': df['time'].iloc[i],
                'type': 'BULLISH_FVG',
                'bottom': df['high'].iloc[i-2],
                'top': df['low'].iloc[i]
            })
        # Bearish FVG
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg_list.append({
                'time': df['time'].iloc[i],
                'type': 'BEARISH_FVG',
                'bottom': df['high'].iloc[i],
                'top': df['low'].iloc[i-2]
            })
    return pd.DataFrame(fvg_list)

def find_order_blocks(df):
    """شناسایی اردر بلاک‌های کلیدی (Order Blocks)"""
    ob_list = []
    for i in range(3, len(df) - 1):
        # Bullish OB
        if df['close'].iloc[i] > df['high'].iloc[i-1] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            ob_list.append({
                'time': df['time'].iloc[i-1],
                'type': 'BULLISH_OB',
                'bottom': df['low'].iloc[i-1],
                'top': df['high'].iloc[i-1]
            })
        # Bearish OB
        elif df['close'].iloc[i] < df['low'].iloc[i-1] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
            ob_list.append({
                'time': df['time'].iloc[i-1],
                'type': 'BEARISH_OB',
                'bottom': df['low'].iloc[i-1],
                'top': df['high'].iloc[i-1]
            })
    return pd.DataFrame(ob_list)

if __name__ == "__main__":
    if mt5.initialize():
        symbol = "XAUUSD"
        print(f"--- تحلیل زنده SMC برای نماد {symbol} ---")
        
        df = fetch_market_data(symbol, mt5.TIMEFRAME_M15, 200)
        
        if df is not None:
            fvgs = find_fair_value_gaps(df)
            obs = find_order_blocks(df)
            
            print(f"\nتعداد FVGهای یافت شده: {len(fvgs)}")
            if not fvgs.empty:
                print("آخرین FVG شناسایی‌شده:")
                print(fvgs.tail(2))
                
            print(f"\nتعداد Order Blockهای یافت شده: {len(obs)}")
            if not obs.empty:
                print("آخرین Order Block شناسایی‌شده:")
                print(obs.tail(2))
                
        mt5.shutdown()