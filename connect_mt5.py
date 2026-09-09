import MetaTrader5 as mt5

def test_connection():
    # 1. راه اندازی و اتصال به متاتریدر 5
    if not mt5.initialize():
        print(f"❌ خطا در اتصال به متاتریدر 5: {mt5.last_error()}")
        return False

    print("🚀 اتصال پایتون به متاتریدر 5 با موفقیت برقرار شد!")

    # 2. دریافت مشخصات حساب زنده/دمو
    account = mt5.account_info()
    if account is not None:
        print(f"📌 شماره حساب: {account.login}")
        print(f"💰 موجودی (Balance): ${account.balance}")
        print(f"سرور بروکر: {account.server}")

    # 3. قطع اتصال
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_connection()