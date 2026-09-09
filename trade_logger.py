import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "bot_trades.db"

def init_db():
    """ایجاد دیتابیس جهت ثبت تاریخچه معاملات ربات"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            entry_price REAL,
            sl REAL,
            tp REAL,
            result TEXT, -- 'WIN', 'LOSS', 'OPEN'
            reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(symbol, action, entry, sl, tp, reason):
    """ثبت معامله جدید در دیتابیس"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO trades (timestamp, symbol, action, entry_price, sl, tp, result, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, symbol, action, entry, sl, tp, 'OPEN', reason))
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()
    return trade_id

def update_trade_result(trade_id, result):
    """به‌روزرسانی نتیجه معامله (WIN یا LOSS)"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE trades SET result = ? WHERE id = ?', (result, trade_id))
    conn.commit()
    conn.close()

def get_recent_results(limit=2):
    """دریافت نتایج آخرین معاملات ثبت‌شده"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT * FROM trades WHERE result != 'OPEN' ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    print("✅ دیتابیس معاملات با موفقیت آماده‌سازی شد.")