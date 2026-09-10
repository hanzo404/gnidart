"""تست‌های موتور بک‌تست — واقع‌گرایی هزینه‌ها، گپ، ضد-نشتی.

هر تست یک رفتار قراردادی موتور را قفل می‌کند؛ اگر بعداً کسی (مثلاً من)
 «بهینه‌سازی»‌ای این رفتارها را عوض کند، همین‌جا لو می‌رود.
"""
import unittest

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestConfig, Backtester, Order
from bot.backtest.v0_strategy import V0Strategy


def make_bars(rows):
    """rows: [(min, open, high, low, close), ...] — اسپرد ثابت $0.10."""
    df = pd.DataFrame(rows, columns=["min", "open", "high", "low", "close"])
    df["time"] = pd.Timestamp("2024-01-02 00:00") + pd.to_timedelta(df.pop("min"), unit="m")
    df["volume"] = 100.0
    df["spread_usd"] = 0.10
    return df


class Scripted:
    """استراتژی آزمایشی: سیگنال فقط روی اندیس‌های داده‌شده."""

    def __init__(self, signals):
        self.signals = signals  # {i: Order}

    def prepare(self, bars):
        pass

    def on_bar(self, i):
        return self.signals.get(i)


def run(bars, signals, **cfg):
    config = BacktestConfig(slippage_usd=0.05, **cfg)
    return Backtester(bars, Scripted(signals), config).run()


class TestExecution(unittest.TestCase):
    def test_fill_next_open_with_spread_no_lookahead(self):
        """سیگنال در close کندل ۱ → پر شدن در open کندل ۲ با ask (+اسپرد+اسلیپیج)."""
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.8, 101.2, 100.7, 101.0),
                          (3, 101.0, 101.5, 100.9, 101.4)])
        res = run(bars, {1: Order(+1, stop=95.0, rr=2.5)})
        self.assertEqual(len(res.trades), 1)
        tr = res.trades.iloc[0]
        self.assertAlmostEqual(tr["entry"], 100.80 + 0.10 + 0.05, places=6)
        self.assertEqual(tr["entry_time"], bars["time"].iloc[2])
        self.assertEqual(tr["reason"], "end_of_data")

    def test_gap_stop_fills_at_open_not_at_stop(self):
        """گپ آخر هفته: استاپ با قیمتِ open بعد از گپ پر می‌شود، نه قیمت استاپ."""
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.5, 100.8, 100.3, 100.6),
                          (3, 94.00, 94.50, 93.50, 94.20)])
        res = run(bars, {1: Order(+1, stop=95.0, rr=2.5)})
        tr = res.trades.iloc[0]
        self.assertEqual(tr["reason"], "gap_stop")
        self.assertAlmostEqual(tr["exit"], 94.00 - 0.05, places=6)
        self.assertLess(tr["pnl"], 0)

    def test_favorable_gap_target_fills_at_open(self):
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.5, 100.8, 100.3, 100.6),
                          (3, 120.0, 120.5, 119.0, 119.5)])
        res = run(bars, {1: Order(+1, stop=95.0, rr=2.5)})
        tr = res.trades.iloc[0]
        self.assertEqual(tr["reason"], "gap_target")
        self.assertAlmostEqual(tr["exit"], 120.0 - 0.05, places=6)

    def test_stop_first_when_both_hit_same_bar(self):
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.5, 100.8, 100.3, 100.6),
                          (3, 100.5, 103.0, 99.0, 100.8)])
        res = run(bars, {1: Order(+1, stop=100.0, rr=2.5)})
        tr = res.trades.iloc[0]
        self.assertEqual(tr["reason"], "stop")
        self.assertAlmostEqual(tr["exit"], 100.0 - 0.05, places=6)

    def test_short_exit_pays_spread_on_ask(self):
        """خروج فروش = خرید با ask → exit = قیمت + اسپرد + اسلیپیج."""
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.4, 100.7, 100.2, 100.5),
                          (3, 96.0, 96.2, 94.8, 95.0)])
        res = run(bars, {1: Order(-1, stop=105.0, rr=1.0)})
        tr = res.trades.iloc[0]
        fill = 100.40 - 0.05          # ورود فروش با bid − اسلیپیج
        target = fill - 1.0 * (105.0 - fill)
        self.assertEqual(tr["reason"], "target")
        self.assertAlmostEqual(tr["exit"], target + 0.10 + 0.05, places=6)
        self.assertAlmostEqual(tr["pnl"], (fill - (target + 0.15)) * 1.0, places=6)

    def test_target_computed_from_fill(self):
        """TP = قیمت واقعی پر شدن + rr × فاصله تا استاپ (نه از قیمت سیگنال)."""
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.5, 110.0, 100.3, 109.0),
                          (3, 109.0, 111.0, 108.5, 110.5)])
        res = run(bars, {1: Order(+1, stop=95.0, rr=2.5)})
        tr = res.trades.iloc[0]
        fill = 100.50 + 0.15
        self.assertAlmostEqual(tr["target"], fill + 2.5 * (fill - 95.0), places=6)


class TestGuards(unittest.TestCase):
    def _rising(self, n=20):
        rows = []
        px = 100.0
        for i in range(n):
            rows.append((i, px, px + 0.4, px - 0.3, px + 0.2))
            px += 0.2
        return make_bars(rows)

    def test_cooldown_blocks_early_entries(self):
        bars = self._rising(20)
        res = run(bars, {1: Order(+1, stop=90.0), 2: Order(+1, stop=90.0),
                         9: Order(+1, stop=90.0)},
                  cooldown_bars=5, max_positions=10)
        # ورودها: کندل ۲ (سیگنال ۱) و کندل ۱۰ (سیگنال ۹؛ فاصله ۸ ≥ ۵)
        # سیگنال ۲ رد می‌شود: کندل ۳ فقط ۱ کندل بعد از ورود قبلی است
        entries = list(res.trades["entry_time"])
        self.assertEqual(entries, [bars["time"].iloc[2], bars["time"].iloc[10]])

    def test_max_positions_one(self):
        bars = self._rising(20)   # روند صعودی → پوزیشن اول باز می‌ماند
        sigs = {i: Order(+1, stop=50.0) for i in range(1, 10)}
        res = run(bars, sigs, max_positions=1)
        self.assertEqual(len(res.trades), 1)

    def test_spread_gate_blocks_entry(self):
        bars = self._rising(10)
        bars.loc[2, "spread_usd"] = 0.99   # لحظهٔ ورود گران
        res = run(bars, {1: Order(+1, stop=90.0)}, spread_gate_usd=0.40)
        self.assertEqual(len(res.trades), 0)

    def test_risk_pct_sizing(self):
        bars = self._rising(10)
        o2 = bars["open"].iloc[2]
        stop = o2 + 0.15 - 5.0            # فاصله دقیقاً $5 از fill
        res = run(bars, {1: Order(+1, stop=stop)}, risk_pct=0.01,
                  min_lots=0.01, lot_step=0.01)
        tr = res.trades.iloc[0]
        # ریسک ۱٪ از ۳۰۰۰ = $30 → 30/(5×100) = 0.06 لات
        self.assertAlmostEqual(tr["lots"], 0.06, places=6)

    def test_halt_on_ruin(self):
        bars = make_bars([(0, 100.0, 100.5, 99.5, 100.2),
                          (1, 100.2, 100.6, 100.1, 100.5),
                          (2, 100.5, 100.8, 100.3, 100.6),
                          (3, 99.0, 99.2, 98.8, 99.0)])
        res = run(bars, {1: Order(+1, stop=95.0)}, fixed_lots=10.0,
                  start_equity=100.0)
        self.assertTrue(res.halted)
        self.assertIn("margin_call", set(res.trades["reason"]))
        self.assertLessEqual(res.metrics["final_equity"], 0)


class TestV0Strategy(unittest.TestCase):
    def test_fvg_detection_and_bias_gate(self):
        """FVG صعودی + بایاس خرسی = بدون سیگنال؛ FVG نزولی + خرسی = فروش."""
        rows = [(i, 100 + i, 100.5 + i, 99.5 + i, 100.2 + i) for i in range(20)]
        rows[10] = (10, 130.0, 131.0, 129.5, 130.5)   # bull FVG در کندل ۱۰
        rows[12] = (12, 70.0, 70.5, 69.5, 70.0)       # bear FVG در کندل ۱۲
        bars = make_bars(rows)
        h4 = pd.DataFrame({
            "time": [pd.Timestamp("2023-12-31") + pd.Timedelta(hours=4 * k) for k in range(18)],
            "open": [200.0] * 18, "high": [200.5] * 18, "low": [199.5] * 18,
            "close": [200 - k for k in range(18)],  # نزولی → BEARISH
        })
        strat = V0Strategy(h4)
        strat.prepare(bars)
        self.assertIsNone(strat.on_bar(10))   # FVG صعودی ولی bias=−1 → ممنوع
        sig = strat.on_bar(12)                # FVG نزولی + bias=−1 → فروش
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, -1)
        self.assertAlmostEqual(sig.stop, 131.0 + 0.50, places=6)

    def test_neutral_bias_allows_both(self):
        rows = [(i, 100.0, 100.5, 99.5, 100.2) for i in range(20)]
        rows[10] = (10, 130.0, 131.0, 129.5, 130.5)  # bull FVG
        bars = make_bars(rows)
        h4 = pd.DataFrame({"time": [pd.Timestamp("2024-01-01 04:00")],
                           "open": [100.0], "high": [100.5],
                           "low": [99.5], "close": [100.2]})  # <10 کندل → NEUTRAL
        strat = V0Strategy(h4)
        strat.prepare(bars)
        sig = strat.on_bar(10)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, +1)
        self.assertAlmostEqual(sig.stop, 99.5 - 0.50, places=6)  # کف ۵ کندل − pad


if __name__ == "__main__":
    unittest.main()
