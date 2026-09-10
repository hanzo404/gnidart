"""تست‌های فاز ۵ — حلقهٔ زنده با Provider/Adapter جعلی (بدون MT5).

همان فلسفهٔ معماری: اگر منطقِ تصمیم درست باشد روی جعبهٔ ساختگی،
روی MT5 واقعی هم درست است — چون runner هیچ چیزی از MT5 نمی‌داند.
"""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import BotConfig
from bot.execution.base import Fill, Order
from bot.journal.store import Journal
from bot.live.runner import LiveRunner, us_dst_active


def uptrend_bars(n=600, seed=7, gap_last=True, start="2026-06-01 12:00",
                freq="15min"):
    """M15 صعودی با FVG صعودی روی آخرین کندلِ بسته."""
    rng = np.random.default_rng(seed)
    step = rng.normal(1.2, 0.6, n)
    open_ = 2000.0 + np.concatenate([[0], np.cumsum(step[:-1])])
    close = open_ + step
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    t = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({"time": t, "open": open_, "high": high, "low": low,
                       "close": close, "tick_volume": 100.0, "spread": 14.0})
    if gap_last:  # FVG صعودی: low[i] > high[i-2]
        i = n - 1
        df.loc[i, "low"] = df.loc[i - 2, "high"] + 1.0
        df.loc[i, "open"] = df.loc[i - 1, "close"] + 0.5
        df.loc[i, "close"] = df.loc[i, "open"] + 1.5
        df.loc[i, "high"] = df.loc[i, "close"] + 0.5
    return df


def h4_bars(n=150, start="2026-05-01 00:00"):
    t = pd.date_range(start, periods=n, freq="4h")
    c = 2000 + np.arange(n) * 0.8
    return pd.DataFrame({"time": t, "open": c - 0.4, "high": c + 0.6,
                         "low": c - 1.0, "close": c, "tick_volume": 500.0,
                         "spread": 14.0})


class FakeProvider:
    def __init__(self, bars, h4):
        self.bars = bars
        self.h4 = h4

    def candles(self, timeframe, count, closed_only=True):
        return (self.bars if timeframe == "M15" else self.h4).tail(count) \
            .reset_index(drop=True)

    def account_summary(self):
        return {"login": 1, "server": "fake", "balance": 3000.0,
                "currency": "USD", "is_demo": True, "leverage": 100}

    def live_quote(self):
        bid = float(self.bars["close"].iloc[-1])
        return {"bid": bid, "ask": bid + 0.14, "point": 0.01,
                "spread_points": 14.0}


class FakeAdapter:
    def __init__(self):
        self.orders = []
        self.positions = []

    def place_order(self, order: Order) -> Fill:
        self.orders.append(order)
        self.positions.append({"id": "T1", "direction": order.direction,
                               "units": order.units, "entry": order.price or 0,
                               "profit": 0.0})
        return Fill(order_id="T1", ts=datetime.now(), price=2000.0,
                    units=order.units)

    def close_position(self, position_id, ts, price):
        self.positions = [p for p in self.positions if p["id"] != position_id]

    def open_positions(self):
        return list(self.positions)

    def close_info(self, position_id):
        return {"exit_price": 1990.0, "profit": -15.0}


def make_runner(tmp, bars=None, h4=None, dry_run=False, **kw):
    provider = FakeProvider(bars if bars is not None else uptrend_bars(),
                            h4 if h4 is not None else h4_bars())
    adapter = FakeAdapter()
    journal = Journal(str(Path(tmp) / "journal.db"))
    cfg = BotConfig()
    runner = LiveRunner(provider, adapter, journal, cfg,
                        state_path=str(Path(tmp) / "state.json"),
                        dry_run=dry_run, utc_offset=180, **kw)
    return runner, provider, adapter, journal


class TestUsDst(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(us_dst_active(datetime(2026, 7, 1)))
        self.assertFalse(us_dst_active(datetime(2026, 1, 1)))
        self.assertTrue(us_dst_active(datetime(2026, 3, 10)))   # دومین یکشنبه مارس = ۸/۳
        self.assertFalse(us_dst_active(datetime(2026, 11, 2)))  # بعد از ۱ نوامبر


class TestEntry(unittest.TestCase):
    def test_signal_places_order_and_journals(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, _, ad, jr = make_runner(tmp)
            msg = r.on_cycle()
            self.assertIn("ورود", msg)
            self.assertEqual(len(ad.orders), 1)
            o = ad.orders[0]
            self.assertEqual(o.direction, 1)
            self.assertLess(o.stop, o.price)          # استاپ زیر ورود
            self.assertGreater(o.target, o.price)     # تارگت بالای ورود
            # TP = ورود + 2.5 × فاصله (همان قرارداد بک‌تست)
            self.assertAlmostEqual(o.target,
                                   o.price + 2.5 * (o.price - o.stop), places=4)
            self.assertGreaterEqual(o.units, 0.01)
            rows = jr.recent_trades(10, status="open")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["regime"], "TREND_UP")
            self.assertEqual(rows[0]["features"]["spread_usd"], 0.14)

    def test_no_double_entry_while_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, _ = make_runner(tmp)
            r.on_cycle()
            prov.bars = uptrend_bars(start="2026-06-01 13:00")  # کندل جدید
            msg = r.on_cycle()
            self.assertEqual(len(ad.orders), 1)
            self.assertIn("پوزیشن باز", msg)

    def test_spread_gate_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, _ = make_runner(tmp)
            q = prov.live_quote()
            prov.live_quote = lambda: {**q, "ask": q["bid"] + 0.99,
                                       "spread_points": 99.0}
            msg = r.on_cycle()
            self.assertEqual(len(ad.orders), 0)
            self.assertIn("اسپرد", msg)

    def test_session_gate_blocks_outside_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            # آخرین کندل بسته: 21:45 سرور تابستان (+3) → ورود 22:00 = 19:00 UTC باز
            bars = uptrend_bars(start="2026-06-01 12:00")
            r, _, ad, _ = make_runner(tmp, bars=bars)
            self.assertIn("ورود", r.on_cycle())
            # ولی 23:45 → ورود 00:00 UTC = خارج از سشن ۱۲–۲۰
            bars2 = uptrend_bars(start="2026-06-01 14:00")
            bars2["time"] = bars2["time"].apply(
                lambda t: t.replace(hour=23, minute=45))
            r2, _, ad2, _ = make_runner(tmp, bars=bars2)
            msg = r2.on_cycle()
            self.assertEqual(len(ad2.orders), 0)
            self.assertIn("سشن", msg)

    def test_direction_filter_blocks_shorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            # کندل نزولی با FVG نزولی آخر
            bars = uptrend_bars(seed=9, gap_last=False)
            n = len(bars)
            desc = bars.iloc[::-1].reset_index(drop=True)
            desc["time"] = bars["time"]
            # گپ نزولی: high[i] < low[i-2]
            desc.loc[n - 1, "high"] = desc.loc[n - 3, "low"] - 1.0
            desc.loc[n - 1, "open"] = desc.loc[n - 2, "close"] - 0.5
            desc.loc[n - 1, "close"] = desc.loc[n - 1, "open"] - 1.5
            desc.loc[n - 1, "low"] = desc.loc[n - 1, "close"] - 0.5
            r, _, ad, _ = make_runner(tmp, bars=desc)
            msg = r.on_cycle()
            # فیلتر پیش‌فرض long → فروش بلاک (اگر اصلاً سیگنال فروش بیاید)
            self.assertNotIn("ورود انجام شد", msg)


class TestPaperLifecycle(unittest.TestCase):
    def test_entry_then_stop_close_and_breaker_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, jr = make_runner(tmp, dry_run=True)
            self.assertIn("ورود", r.on_cycle())
            p = r.paper
            self.assertIsNotNone(p)
            # کندل بعدی: سقوط به زیر استاپ
            nxt = uptrend_bars(seed=3, gap_last=False, start="2026-06-02 12:00")
            nxt.loc[len(nxt) - 1, "low"] = p.stop - 2.0
            nxt.loc[len(nxt) - 1, "close"] = p.stop - 1.0
            prov.bars = nxt
            r.on_cycle()
            self.assertIsNone(r.paper)
            closed = jr.recent_trades(10, status="closed")
            self.assertEqual(len(closed), 1)
            self.assertLess(closed[0]["r_multiple"], 0)
            self.assertGreaterEqual(r.breaker.streak, 1)

    def test_breaker_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = str(Path(tmp) / "state.json")
            provider = FakeProvider(uptrend_bars(), h4_bars())
            journal = Journal(str(Path(tmp) / "j.db"))
            cfg = BotConfig()
            r1 = LiveRunner(provider, FakeAdapter(), journal, cfg,
                            state_path=state, dry_run=True, utc_offset=180)
            for k in range(3):
                r1.breaker.on_trade_closed(-1.0, datetime(2026, 6, 1, 10 + k))
            r1._save_state()
            self.assertAlmostEqual(r1.breaker.size_multiplier(), 0.5)
            # ری‌استارت: وضعیت بریکر از دیسک برمی‌گردد
            r2 = LiveRunner(FakeProvider(uptrend_bars(), h4_bars()),
                            FakeAdapter(), Journal(str(Path(tmp) / "j.db")),
                            cfg, state_path=state, dry_run=True,
                            utc_offset=180)
            self.assertAlmostEqual(r2.breaker.size_multiplier(), 0.5)
            self.assertEqual(r2.breaker.streak, 3)


if __name__ == "__main__":
    unittest.main()
