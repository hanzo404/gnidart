"""تست پاریتی بک‌تست ↔ لایو — v0.5.8 (آیتم ۶ اسپرینت).

فلسفه: همان داده از دو مسیر باید همان معامله بدهد —
  مسیر A: موتور بک‌تست event-driven (Backtester + V0Strategy)
  مسیر B: حلقهٔ زندهٔ on_cycle با Provider بازپخش‌شونده (dry_run/کاغذی)

این تست «نشت معنایی» بین دو مسیر را در CI می‌گیرد — همان خانوادهٔ
باگی که در فاز ۵ دیدیم (retry داخل کندل / ورود مجدد بعد از استاپِ همان
کندل): هر دو از یک باگ مشترکِ منطق رنج می‌برند، اما وقتی یک مسیر
تکامل می‌کند و دیگری نه، این تست قرمز می‌شود.

هم‌ترازی‌های عمدی برای پاریتی دقیق:
  - open کندلِ بعد از FVG = close کندل FVG → قیمت ورود دو مسیر یکی
  - spread_usd ستون داده = point×spread_points کوتیت جعلی
  - slippage_usd=0 و be_at_frac=None در بک‌تست (کاغذی این‌ها را ندارد)
  - utc_offset ثابت 180 (سشن‌ها یکسان قطع/وصل می‌شوند)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest.engine import Backtester, BacktestConfig
from bot.backtest.risk import BreakerPolicy
from bot.backtest.v0_strategy import V0Strategy
from bot.config import BotConfig
from bot.journal.store import Journal
from bot.live.runner import LiveRunner
from bot.risk.circuit_breaker import CircuitBreaker
from tests.test_live_runner import FakeAdapter, h4_bars

FVG_K = 400          # کندل سیگنال (FVG صعودی)
TAIL = 13            # کندل‌های بعد از سیگنال (خروج باید داخل این‌ها رخ دهد)
SCALE = 2.0          # مقیاس قیمتی — استاپِ دورتر → لات کوچک‌تر از سقف دمو


def parity_bars() -> pd.DataFrame:
    """M15 صعودیِ آرام (بدون FVG طبیعی) + یک FVG تزریقی روی کندل FVG_K.

    سه بخش:
      1) روندِ ملایمِ پیوسته (شیب 0.1×SCALE + نویز کم) — رژیم صعودی می‌دهد
         ولی هیچ FVG طبیعی نمی‌سازد (شرط FVG: گام > 2×سایه — برقرار نمی‌شود)
      2) FVG صعودی بزرگ روی کندل k + کندل پیوستگی (open بعدی = close)
      3) سقوط کنترل‌شده (۲$ در هر کندل) → خروجِ «stop» داخل کندل، بدون گپ
         از استاپ رد نمی‌شود (قیمت خروج دو مسیر دقیقاً یکی می‌ماند)
    """
    n = FVG_K + TAIL
    rng = np.random.default_rng(7)
    step = 0.10 * SCALE + rng.normal(0, 0.03 * SCALE, n)
    t = pd.date_range("2026-06-01 12:00", periods=n, freq="15min")
    open_ = 2000.0 + np.concatenate([[0.0], np.cumsum(step[:-1])])
    close = open_ + step
    w = 0.5 * SCALE
    high = np.maximum(open_, close) + w
    low = np.minimum(open_, close) - w

    # --- ۲) FVG تزریقی روی کندل k ---
    k = FVG_K
    low[k] = high[k - 2] + 1.0 * SCALE
    open_[k] = close[k - 1] + 0.5 * SCALE
    close[k] = open_[k] + 1.5 * SCALE
    high[k] = close[k] + 0.5 * SCALE
    # پیوستگی: open کندل بعد = close این کندل (شرط برابری قیمت ورود دو مسیر)
    open_[k + 1] = close[k]
    close[k + 1] = open_[k + 1] + 0.1
    high[k + 1] = max(open_[k + 1], close[k + 1]) + w
    low[k + 1] = min(open_[k + 1], close[k + 1]) - w
    # --- 3) سقوط کنترل‌شده تا خوردن استاپ، بعد آرامش ---
    for j in range(k + 2, k + 6):          # ۴ کندل −2$ (بدون گپ از استاپ)
        open_[j] = close[j - 1]
        close[j] = open_[j] - 2.0
        high[j] = open_[j] + 0.2
        low[j] = close[j] - 0.2
    for j in range(k + 6, n):              # دنبالهٔ بی‌سیگنال
        open_[j] = close[j - 1]
        close[j] = open_[j] - 0.05
        high[j] = open_[j] + 0.2
        low[j] = close[j] - 0.2

    df = pd.DataFrame({"time": t, "open": open_, "high": high, "low": low,
                       "close": close, "tick_volume": 100.0, "spread": 14.0})
    # ستون اسپرد دلاری — همان چیزی که کوتیت جعلی برمی‌گرداند
    df["spread_usd"] = 0.01 * 14.0
    return df


class ReplayProvider:
    """Provider جعلی که تاریخچه را «تدریجی» بازپخش می‌کند (کندل به کندل).

    تفاوتش با FakeProvider فاز ۵: هر سیکل فقط پیشوندِ bars[:c] را
    می‌بیند — دقیقاً مثل اینکه لایو روی جریان زندهٔ دیتا اجرا شود.
    """

    def __init__(self, bars, h4, point=0.01, spread_points=14.0,
                 balance=3000.0):
        self.bars = bars
        self.h4 = h4
        self.point = point
        self.spread_points = spread_points
        self.balance = balance
        self.c = 0                     # طول پیشوند فعلی

    def candles(self, timeframe, count, closed_only=True):
        src = self.bars if timeframe == "M15" else self.h4
        return src.iloc[:self.c].tail(count).reset_index(drop=True)

    def account_summary(self):
        return {"login": 1, "server": "replay", "balance": self.balance,
                "currency": "USD", "is_demo": True, "leverage": 100}

    def live_quote(self):
        bid = float(self.bars["close"].iloc[self.c - 1])
        return {"bid": bid, "ask": bid + self.point * self.spread_points,
                "point": self.point, "spread_points": self.spread_points}


class TestBacktestLiveParity(unittest.TestCase):
    def test_same_data_same_trade(self):
        bars = parity_bars()
        h4 = h4_bars()
        n = len(bars)

        # ---------- مسیر A: بک‌تست event-driven ----------
        cfg_bt = BacktestConfig(
            start_equity=3000.0, risk_pct=0.005,
            contract_oz=100.0,          # پروفایل طلا
            min_lots=0.01, lot_step=0.01,
            slippage_usd=0.0,           # کاغذی اسلیپیج لاگ نمی‌کند
            spread_gate_usd=None,       # گیت اسپرد لایو خودش دارد (0.40)
            max_positions=1, cooldown_bars=0, be_at_frac=None)
        breaker_bt = CircuitBreaker(
            derate_at=3, deep_derate_at=4, pause_at=5, halt_at=6,
            pause_hours=24.0, max_monthly_dd=0.06)
        bt = Backtester(bars, V0Strategy(h4, sl_pad=0.50, rr=2.5),
                        config=cfg_bt, risk_policy=BreakerPolicy(breaker_bt))
        res = bt.run()
        bt_trades = res.trades

        # سلامت فیکسچر: دقیقاً یک معامله، با خروج واقعی (نه پایان-داده)
        self.assertEqual(len(bt_trades), 1,
                         f"فیکسچر باید دقیقاً یک معامله بدهد؛ شد: "
                         f"{len(bt_trades)}")
        self.assertNotEqual(bt_trades.iloc[0]["reason"], "end_of_data",
                            "خروج باید داخل داده رخ دهد — TAIL را زیاد کن")
        t_bt = bt_trades.iloc[0]

        # ---------- مسیر B: بازپخش از حلقهٔ on_cycle ----------
        with tempfile.TemporaryDirectory() as tmp:
            provider = ReplayProvider(bars, h4)
            journal = Journal(str(Path(tmp) / "journal.db"))
            cfg = BotConfig()
            runner = LiveRunner(provider, FakeAdapter(), journal, cfg,
                                state_path=str(Path(tmp) / "state.json"),
                                dry_run=True, utc_offset=180,
                                print_fn=lambda *a, **k: None)

            live_entries = []   # {bar_time, direction, price}
            live_exits = []     # {bar_time, price, reason, r}
            orig_close = runner._close_paper

            def _spy_close(price, reason):
                p = runner.paper
                profit = ((price - p.entry) if p.direction > 0
                          else (p.entry - price)) * p.units * 100.0
                r = profit / p.risk_usd if p.risk_usd else 0.0
                live_exits.append({
                    "bar_time": str(provider.bars["time"]
                                    .iloc[provider.c - 1]),
                    "price": price, "reason": reason, "r": r})
                return orig_close(price, reason)

            runner._close_paper = _spy_close

            for c in range(260, n + 1):        # گرم‌کردن ۲۵۰+ کندل
                provider.c = c
                runner.on_cycle()
                if runner.paper is not None and not live_entries:
                    live_entries.append({
                        "bar_time": str(provider.bars["time"].iloc[c - 1]),
                        "direction": runner.paper.direction,
                        "price": runner.paper.entry})

            # ---------- مقایسه (داخل tmp — ژورنال هنوز زنده است) ----------
            self.assertEqual(len(live_entries), 1,
                             "لایو باید دقیقاً یک ورود بگیرد")
            self.assertEqual(len(live_exits), 1,
                             "لایو باید دقیقاً یک خروج بگیرد")
            e, x = live_entries[0], live_exits[0]

            # ورود: بک‌تست در open کندلِ «بعد از» سیگنال پر می‌شود؛
            # لایو همان لحظهٔ بسته‌شدن کندل سیگنال (قیمت یکی، برچسب ۱۵ دقیقه جابه‌جا)
            entry_bt = pd.Timestamp(t_bt["entry_time"])
            entry_live = pd.Timestamp(e["bar_time"]) + pd.Timedelta(minutes=15)
            self.assertEqual(entry_bt, entry_live,
                             "کندل ورود دو مسیر باید یکی باشد")
            self.assertEqual(int(t_bt["direction"]), int(e["direction"]))
            self.assertAlmostEqual(float(t_bt["entry"]), float(e["price"]),
                                   places=6, msg="قیمت ورود دو مسیر")

            # خروج: همان کندل، همان قیمت، همان دلیل
            self.assertEqual(pd.Timestamp(t_bt["exit_time"]),
                             pd.Timestamp(x["bar_time"]), "کندل خروج")
            self.assertAlmostEqual(float(t_bt["exit"]), float(x["price"]),
                                   places=6, msg="قیمت خروج دو مسیر")
            self.assertEqual(t_bt["reason"], x["reason"], "دلیل خروج")

            # R یکسان = کل زنجیرهٔ سایز/ریسک هم‌تراز است
            self.assertAlmostEqual(float(t_bt["r"]), x["r"],
                                   places=6, msg="R دو مسیر")


if __name__ == "__main__":
    unittest.main()
