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
                freq="15min", scale=1.0):
    """M15 صعودی با FVG صعودی روی آخرین کندلِ بسته.

    scale=1.0 → مقیاس طلا (فیکسچرهای فاز ۵، دست‌نخورده)
    scale=0.015 → مقیاس نقره (~$30)؛ scale=0.06 → نقره با استاپ دور (تست skip)
    """
    rng = np.random.default_rng(seed)
    step = rng.normal(1.2, 0.6, n) * scale
    open_ = 2000.0 + np.concatenate([[0], np.cumsum(step[:-1])])
    close = open_ + step
    high = np.maximum(open_, close) + 0.5 * scale
    low = np.minimum(open_, close) - 0.5 * scale
    t = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({"time": t, "open": open_, "high": high, "low": low,
                       "close": close, "tick_volume": 100.0, "spread": 14.0})
    if gap_last:  # FVG صعودی: low[i] > high[i-2]
        i = n - 1
        df.loc[i, "low"] = df.loc[i - 2, "high"] + 1.0 * scale
        df.loc[i, "open"] = df.loc[i - 1, "close"] + 0.5 * scale
        df.loc[i, "close"] = df.loc[i, "open"] + 1.5 * scale
        df.loc[i, "high"] = df.loc[i, "close"] + 0.5 * scale
    return df


def h4_bars(n=150, start="2026-05-01 00:00"):
    t = pd.date_range(start, periods=n, freq="4h")
    c = 2000 + np.arange(n) * 0.8
    return pd.DataFrame({"time": t, "open": c - 0.4, "high": c + 0.6,
                         "low": c - 1.0, "close": c, "tick_volume": 500.0,
                         "spread": 14.0})


class FakeProvider:
    def __init__(self, bars, h4, point=0.01, spread_points=14.0):
        self.bars = bars
        self.h4 = h4
        self.point = point
        self.spread_points = spread_points

    def candles(self, timeframe, count, closed_only=True):
        return (self.bars if timeframe == "M15" else self.h4).tail(count) \
            .reset_index(drop=True)

    def account_summary(self):
        return {"login": 1, "server": "fake", "balance": 3000.0,
                "currency": "USD", "is_demo": True, "leverage": 100}

    def live_quote(self):
        bid = float(self.bars["close"].iloc[-1])
        return {"bid": bid, "ask": bid + self.point * self.spread_points,
                "point": self.point, "spread_points": self.spread_points}


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


def make_runner(tmp, bars=None, h4=None, dry_run=False, point=0.01,
                spread_points=14.0, **kw):
    provider = FakeProvider(bars if bars is not None else uptrend_bars(),
                            h4 if h4 is not None else h4_bars(),
                            point=point, spread_points=spread_points)
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


# ═══════════════════ فاز ۷: چند-نمادی (طلا + نقره) ═══════════════════
class TestSymbolProfiles(unittest.TestCase):
    def test_gold_profile_untouched(self):
        """رفتار آستین زندهٔ طلا دقیقاً مثل فاز ۵ می‌ماند."""
        g = BotConfig().profile_for("XAUUSD")
        self.assertEqual(g.contract_size, 100.0)
        self.assertEqual(g.sl_pad, 0.50)
        self.assertEqual(g.max_spread_usd, 0.40)
        self.assertEqual(g.max_lots, 0.10)
        self.assertEqual(g.min_lot_policy, "force")
        self.assertEqual(g.digits, 2)
        self.assertAlmostEqual(g.baseline_wr, 0.348)

    def test_silver_profile(self):
        s = BotConfig().profile_for("XAGUSD")
        self.assertEqual(s.contract_size, 5000.0)
        self.assertAlmostEqual(s.sl_pad, 0.01093)
        self.assertEqual(s.max_spread_usd, 0.06)
        self.assertEqual(s.max_lots, 0.05)
        self.assertEqual(s.min_lot_policy, "skip")
        self.assertEqual(s.digits, 3)
        self.assertAlmostEqual(s.baseline_wr, 0.308)

    def test_yaml_profiles_load(self):
        cfg = BotConfig.load(Path(__file__).parents[1] / "config" / "config.yaml")
        self.assertEqual(cfg.profile_for("XAGUSD").contract_size, 5000.0)
        self.assertEqual(cfg.profile_for("XAUUSD").sl_pad, 0.50)


class TestSilverSleeve(unittest.TestCase):
    """آستین نقره: سایزینگ با قرارداد ۵۰۰۰ اونسی، گیت اسپرد، سیاست skip."""

    def _silver(self, tmp, bars=None, dry_run=True, **kw):
        kw.setdefault("point", 0.001)
        kw.setdefault("spread_points", 18.0)     # $0.018 — اندازه‌گیری دمو
        bars = bars if bars is not None else uptrend_bars(scale=0.015)
        r, prov, ad, jr = make_runner(tmp, bars=bars, dry_run=dry_run,
                                      symbol="XAGUSD", **kw)
        return r, prov, ad, jr

    def test_silver_entry_sizing_and_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, jr = self._silver(tmp)
            self.assertIn("ورود", r.on_cycle())
            p = r.paper
            self.assertIsNotNone(p)
            # بودجه = 0.5% × $3000 = $15؛ ریسک واقعی = dist × لات × 5000
            self.assertGreaterEqual(p.units, 0.01)
            self.assertLessEqual(p.units, 0.05)          # سقف نقره
            self.assertLessEqual(p.risk_usd, 15.0)       # هرگز بیشتر از بودجه
            rows = jr.recent_trades(10, status="open")
            self.assertEqual(rows[0]["symbol"], "XAGUSD")
            self.assertAlmostEqual(rows[0]["features"]["spread_usd"],
                                   0.018, places=3)
            # استاپ با نقره‌ای‌شده: pad $0.01093 نه $0.50
            self.assertLess(p.stop, p.entry)

    def test_silver_spread_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, _ = self._silver(tmp)
            q = prov.live_quote()
            prov.live_quote = lambda: {**q, "spread_points": 80.0}  # $0.08
            msg = r.on_cycle()
            self.assertEqual(len(ad.orders), 0)
            self.assertIn("اسپرد", msg)

    def test_silver_min_lot_skip(self):
        """استاپِ دور → حجم < ۰.۰۱ لات → نقره معامله را رد می‌کند (طلا force داشت)."""
        with tempfile.TemporaryDirectory() as tmp:
            bars = uptrend_bars(scale=0.06)   # دامنهٔ ۵ کندل ~$0.5 → حجم < حداقل
            r, prov, ad, jr = self._silver(tmp, bars=bars)
            msg = r.on_cycle()
            self.assertEqual(len(ad.orders), 0)
            self.assertIn("رد", msg)
            self.assertEqual(len(jr.recent_trades(10, status="open")), 0)

    def test_silver_paper_close_and_sleeve_pnl(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, jr = self._silver(tmp)
            self.assertIn("ورود", r.on_cycle())
            risk = r.paper.risk_usd
            # بستن روی تارگت: سود = 2.5R با قرارداد ۵۰۰۰
            nxt = uptrend_bars(seed=3, gap_last=False, scale=0.015,
                               start="2026-06-02 12:00")
            nxt.loc[len(nxt) - 1, "high"] = r.paper.target + 0.01
            nxt.loc[len(nxt) - 1, "close"] = r.paper.target
            prov.bars = nxt
            r.on_cycle()
            closed = jr.recent_trades(10, status="closed")
            self.assertEqual(len(closed), 1)
            self.assertAlmostEqual(closed[0]["r_multiple"], 2.5, places=1)
            # سود انباشتهٔ آستین = +2.5R (مبنای بریکر ماهانهٔ per-sleeve)
            self.assertAlmostEqual(r.state["sleeve_pnl"], 2.5 * risk, places=3)

    def test_sleeve_equity_not_raw_balance(self):
        """بریکر ماهانه باید سودِ همین نماد را ببیند نه بالانس کل اکانت."""
        with tempfile.TemporaryDirectory() as tmp:
            r, prov, ad, jr = self._silver(tmp)
            self.assertIn("ورود", r.on_cycle())
            risk = r.paper.risk_usd
            nxt = uptrend_bars(seed=3, gap_last=False, scale=0.015,
                               start="2026-06-02 12:00")
            nxt.loc[len(nxt) - 1, "low"] = r.paper.stop - 0.01
            prov.bars = nxt
            r.on_cycle()
            # چرخهٔ بعدی (بدون سیگنال): equity ثبت‌شده = 3000 − risk
            prov.bars = uptrend_bars(seed=5, gap_last=False, scale=0.015,
                                     start="2026-06-03 12:00")
            r.on_cycle()
            rows = jr.conn.execute(
                "SELECT equity, symbol FROM equity ORDER BY ts").fetchall()
            self.assertAlmostEqual(rows[-1]["equity"], 3000.0 - risk,
                                   places=3)
            self.assertEqual(rows[-1]["symbol"], "XAGUSD")


class TestJournalMigration(unittest.TestCase):
    """DB فاز ۵ (بدون ستون symbol) باید بی‌دردسر ارتقا یابد و داده بماند."""

    def test_old_db_gains_symbol_columns_and_wal(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "old.db")
            c = sqlite3.connect(db)
            c.execute("CREATE TABLE equity ("
                      "ts TEXT NOT NULL, equity REAL NOT NULL)")
            c.execute("CREATE TABLE breaker_events ("
                      "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
                      " kind TEXT NOT NULL, detail TEXT, streak INTEGER,"
                      " size_multiplier REAL)")
            c.execute("INSERT INTO equity VALUES ('2026-01-01T00:00:00', 3000.0)")
            c.commit()
            c.close()
            j = Journal(db)   # مهاجرت خودکار
            mode = j.conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), "wal")   # دو پروسهٔ همزمان
            j.record_equity(datetime.now(), 2990.0, symbol="XAGUSD")
            rows = j.conn.execute(
                "SELECT symbol, equity FROM equity ORDER BY ts").fetchall()
            self.assertIsNone(rows[0]["symbol"])    # ردیف قدیمی دست‌نخورده
            self.assertEqual(rows[1]["symbol"], "XAGUSD")
            j.record_breaker_event(datetime.now(), "derate", "تست", 3, 0.5,
                                   symbol="XAGUSD")
            brk = j.conn.execute(
                "SELECT symbol FROM breaker_events").fetchall()
            self.assertEqual(brk[0]["symbol"], "XAGUSD")


if __name__ == "__main__":
    unittest.main()