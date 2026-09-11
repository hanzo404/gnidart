"""هستهٔ اجرای فوروارد — فاز ۵.

معماری: این کلاس هیچ چیز دربارهٔ MT5 نمی‌داند. داده و اجرا از بیرون
تزریق می‌شوند (Provider/Adapter) — پس کل حلقه در سندباکس لینوکس با
جعبهٔ ساختگی تست می‌شود و روی ویندوزِ تو با MT5 واقعی اجرا می‌گردد.

چرخهٔ هر کندلِ بسته‌شدهٔ M15 (دقیقاً مثل بک‌تست، هیچ چیزی بیشتر):
    ۱. معاملهٔ باز را بررسی کن (بسته شده؟ → ژورنال + بریکر + تشخیص)
    ۲. چک ماهانهٔ بریکر (−۶٪) با equity جاری
    ۳. سیگنال: رژیم → گیت (جهت/رژیم/سشن/اسپرد/بریکر/سقف روزانه) → سایز
    ۴. سفارش market با SL/TP چسبیده + ژورنال + ذخیرهٔ وضعیت

بقا و ازسرگیری: وضعیت بریکر/معامله در JSON ذخیره می‌شود؛ ری‌استارت
ربات چیزی از یاد نمی‌برد. استاپ و تارگت سروری‌اند — اگر ربات
خاموش باشد، بروکر خودش اجرایشان می‌کند.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from bot.analysis.diagnostics import VERDICT_FA, diagnose_streak
from bot.backtest.gate import GatedStrategy
from bot.backtest.risk import BreakerPolicy
from bot.backtest.v0_strategy import V0Strategy
from bot.config import BotConfig, SymbolProfile
from bot.execution.base import ExecutionAdapter, Fill, Order
from bot.journal.store import Journal
from bot.regime.engine import NAMES as REGIME_NAMES
from bot.regime.engine import CHAOS, RegimeEngine, TREND_DOWN, TREND_UP, session_open
from bot.risk.circuit_breaker import CircuitBreaker


# ------------------------------------------------------------------ #
def us_dst_active(dt: datetime) -> bool:
    """قاعدهٔ آمریکایی: دومین یکشنبهٔ مارس ← اولین یکشنبهٔ نوامبر."""
    def nth_sunday(year, month, n):
        d = datetime(year, month, 1)
        first_sun = 1 + (6 - d.weekday()) % 7     # یکشنبه = 6
        return first_sun + 7 * (n - 1)
    y = dt.year
    start = datetime(y, 3, nth_sunday(y, 3, 2))
    end = datetime(y, 11, nth_sunday(y, 11, 1))
    return start <= dt < end


def server_to_utc(ts_server: pd.Timestamp, offset_minutes: int) -> pd.Timestamp:
    return ts_server - pd.Timedelta(minutes=offset_minutes)


@dataclass
class PaperPosition:
    """پوزیشن مجازی حالت dry-run — همان قرارداد open_positions آداپتور."""
    id: str
    direction: int
    units: float
    entry: float
    stop: float
    target: float
    risk_usd: float


class LiveRunner:
    def __init__(self, provider, adapter: ExecutionAdapter, journal: Journal,
                 cfg: BotConfig, state_path: str = "data/live_state.json",
                 symbol: str = "XAUUSD", dry_run: bool = False,
                 utc_offset: Optional[int] = None, print_fn=print):
        self.provider = provider
        self.adapter = adapter
        self.journal = journal
        self.cfg = cfg
        self.symbol = symbol
        # فاز ۷: پارامترهای دلاری از پروفایل نماد (طلا/نقره/…) — نه هاردکد
        self.profile = (cfg.profile_for(symbol)
                        if hasattr(cfg, "profile_for") else SymbolProfile())
        if symbol not in getattr(cfg, "symbol_profiles", {}):
            print_fn(f"⚠️ پروفایل {symbol} در config تعریف نشده — پیش‌فرض طلا اعمال شد")
        self.dry_run = dry_run
        self.state_path = pathlib.Path(state_path)
        self._print = print_fn
        self.paper: Optional[PaperPosition] = None
        self.state: dict = {}
        self._load_state()

        self.breaker = CircuitBreaker(
            derate_at=cfg.breaker.derate_at,
            deep_derate_at=cfg.breaker.deep_derate_at,
            pause_at=cfg.breaker.pause_at, halt_at=cfg.breaker.halt_at,
            pause_hours=cfg.breaker.pause_hours,
            max_monthly_dd=cfg.risk.max_monthly_loss)
        self._restore_breaker()
        self.policy = BreakerPolicy(self.breaker)
        self.utc_offset = utc_offset  # None → قاعدهٔ EET/EEST خودکار
        self.direction_map = {"long": 1, "short": -1, "both": None}
        # جهت: پروفایل نماد مقدم است؛ نبود → فیلتر سراسری live
        direction = (self.profile.direction
                     or getattr(cfg.live, "direction_filter", "both"))
        self.direction = self.direction_map.get(str(direction).lower(), None)

    # ------------------------------------------------------------------ #
    # وضعیت: ذخیره/بارگذاری
    # ------------------------------------------------------------------ #
    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                self.state = {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["breaker"] = self.breaker.state()
        self.state["paper"] = (self.paper.__dict__ if self.paper else None)
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1),
            encoding="utf-8")

    def _restore_breaker(self) -> None:
        b = self.state.get("breaker") or {}
        # بازسازی از JSON تاریخ‌ها
        pu = b.get("paused_until")
        self.breaker._streak = int(b.get("streak", 0))
        self.breaker._level = int(b.get("level", 0))
        self.breaker._halted = bool(b.get("halted", False))
        self.breaker._halt_reason = b.get("halt_reason") or ""
        self.breaker._paused_until = (datetime.fromisoformat(pu)
                                      if pu else None)
        pk = self.state.get("paper")
        if pk:
            self.paper = PaperPosition(**pk)

    def acknowledge(self) -> None:
        ev = self.breaker.acknowledge(datetime.now())
        if ev:
            self.journal.record_breaker_event(
                datetime.now(), "ack", ev.detail, self.breaker.streak,
                self.breaker.size_multiplier(), symbol=self.symbol)
        self._save_state()

    # ------------------------------------------------------------------ #
    def _offset_now(self, ts_server: pd.Timestamp) -> int:
        if self.utc_offset is not None:
            return self.utc_offset
        return 180 if us_dst_active(datetime.now()) else 120

    # ------------------------------------------------------------------ #
    def on_cycle(self) -> str:
        """یک چرخه؛ خروجی: توضیح کوتاه برای لاگ."""
        bars = self.provider.candles("M15", self.cfg.live.history_bars,
                                     closed_only=True)
        if len(bars) < 250:
            return "دادهٔ کافی برای گرم‌کردن اندیکاتورها نیست"
        h4 = self.provider.candles("H4", 150, closed_only=True)
        # نکتهٔ pandas 3.x: Timestamp == str دیگر True نیست → هر دو طرف str
        # (باگ فاز ۵: شرط هیچ‌وقت True نمی‌شد و کل مسیر سیگنال هر poll تکرار
        #  می‌شد؛ ضد-بک‌تست: retry داخل کندل + ورود مجدد بعد از استاپِ همان کندل)
        new_bar = str(bars["time"].iloc[-1])
        if new_bar == self.state.get("last_bar_time"):
            return "کندل جدیدی بسته نشده"
        self.state["last_bar_time"] = new_bar

        now_server = pd.Timestamp(new_bar) + pd.Timedelta(minutes=15)
        off = self._offset_now(now_server)
        now_utc = server_to_utc(now_server, off)
        entry_utc = now_utc  # ورود در open کندل بعدی

        # ---- ۱) معاملهٔ باز ------------------------------------------
        self._manage_position(bars, now_server)

        # ---- ۲) چک ماهانهٔ بریکر --------------------------------------
        # فاز ۷: «سرمایهٔ این آستین» = مبنای اولین چرخه + سود انباشتهٔ همین نماد.
        # (بالانس اکانت مشترک است؛ اگر آن را می‌دادیم، ضررِ نقره بریکرِ طلا را می‌کشید)
        try:
            bal = self.provider.account_summary()
            balance = float(bal.get("balance", 0.0))
            if self.state.get("sleeve_base") is None:
                self.state["sleeve_base"] = balance   # اولین اجرای این آستین
            equity = (float(self.state["sleeve_base"])
                      + float(self.state.get("sleeve_pnl", 0.0)))
            self.journal.record_equity(datetime.now(), equity,
                                       symbol=self.symbol)
            self.policy.on_bar_close(datetime.now(), equity)
            if self.breaker.halted:
                self._print("⛔ بریکر HALT فعال است — ورودی جدید نمی‌گذارم. "
                            "بعد از بازبینی: py scripts/run_live.py --ack")
                self._save_state()
                return "halt"
        except Exception as e:  # noqa: BLE001 — اکانت موقتاً در دسترس نیست
            return f"هشدار: account_summary ناموفق ({e})"

        # ---- ۳) سیگنال و گیت‌ها ---------------------------------------
        action = self._maybe_enter(bars, h4, entry_utc, now_server, equity)
        self._save_state()
        return action

    # ------------------------------------------------------------------ #
    def _manage_position(self, bars: pd.DataFrame,
                         now_server: pd.Timestamp) -> None:
        last = bars.iloc[-1]
        # حالت کاغذی: خروج با H/L کندل بسته‌شده (همان معنای بک‌تست)
        if self.dry_run and self.paper is not None:
            p = self.paper
            if p.direction > 0:
                if last["low"] <= p.stop:
                    self._close_paper(p.stop, "stop")
                    return
                if last["high"] >= p.target:
                    self._close_paper(p.target, "target")
                    return
            else:
                if last["high"] >= p.stop:
                    self._close_paper(p.stop, "stop")
                    return
                if last["low"] <= p.target:
                    self._close_paper(p.target, "target")
                    return
            return
        if self.dry_run or not self.state.get("ticket"):
            return
        # حالت واقعی: اگر تیکت دیگر باز نیست → بسته شده
        open_ids = [p["id"] for p in self.adapter.open_positions()]
        if self.state["ticket"] not in open_ids:
            info = self.adapter.close_info(self.state["ticket"])
            profit = float(info.get("profit") or 0.0)
            r = (profit / self.state["risk_usd"]
                 if self.state.get("risk_usd") else 0.0)
            self._on_trade_closed(profit, r, now_server,
                                  info.get("exit_price") or 0.0)

    def _close_paper(self, price: float, reason: str) -> None:
        p = self.paper
        profit = ((price - p.entry) if p.direction > 0
                  else (p.entry - price)) * p.units * self.profile.contract_size
        r = profit / p.risk_usd if p.risk_usd else 0.0
        self.paper = None
        self._on_trade_closed(profit, r, datetime.now(), price, reason)

    def _on_trade_closed(self, profit: float, r: float,
                         ts, exit_price: float, reason: str = "broker") -> None:
        tid = self.state.pop("trade_id", None)
        self.state.pop("ticket", None)
        self.state.pop("risk_usd", None)
        # سود انباشتهٔ همین آستین — پایهٔ بریکر ماهانهٔ per-sleeve
        self.state["sleeve_pnl"] = (float(self.state.get("sleeve_pnl", 0.0))
                                    + float(profit))
        if tid is not None:
            self.journal.close_trade(tid, ts, float(exit_price), float(r))
        events = self.policy.on_trade_closed(r, ts, profit) or []
        for ev in events:
            self.journal.record_breaker_event(
                ts, ev.kind, ev.detail, ev.streak, ev.size_multiplier,
                symbol=self.symbol)
            self._print(f"⚡ بریکر: {ev.kind} — {ev.detail}")
        # بعد از هر derate/deep_derate: گزارش تشخیصی (لایهٔ فاز ۴)
        if any(ev.kind in ("derate", "deep_derate", "pause", "halt",
                           "monthly_halt") for ev in events):
            self._run_diagnostics(ts)

    def _run_diagnostics(self, ts) -> None:
        try:
            rows = self.journal.recent_trades(200)
            if not rows:
                return
            df = pd.DataFrame(rows)
            df = df.rename(columns={"opened_at": "entry_time",
                                    "closed_at": "exit_time"})
            df["pnl"] = df["r_multiple"].astype(float)   # علامت‌ها هم‌معنا
            df["spread_usd"] = df["features"].apply(
                lambda f: float((f or {}).get("spread_usd", 0.0)))
            bars = self.provider.candles("M15", 600, closed_only=True)
            reg = RegimeEngine().compute(bars)
            rep = diagnose_streak(df, streak=max(self.breaker.streak, 3),
                                  regime_df=reg,
                                  baseline_wr=self.profile.baseline_wr, ts=ts)
            self._print(f"🔬 تشخیص: {VERDICT_FA[rep.verdict]} "
                        f"(اطمینان: {rep.confidence})")
            for f in rep.findings:
                self._print(f"   • {f}")
            self.journal.record_breaker_event(
                ts, "diagnostic", f"{rep.verdict}: {' | '.join(rep.findings)}",
                rep.streak, self.breaker.size_multiplier(), symbol=self.symbol)
        except Exception as e:  # noqa: BLE001 — تشخیص هرگز نباید حلقه را بکشد
            self._print(f"(تشخیص ناموفق: {e})")

    # ------------------------------------------------------------------ #
    def _maybe_enter(self, bars: pd.DataFrame, h4: pd.DataFrame,
                     entry_utc: pd.Timestamp, now_server, equity: float) -> str:
        if self.paper is not None:
            return "پوزیشن باز داریم (سقف ۱)"
        if not self.dry_run:
            if self.adapter.open_positions():
                return "پوزیشن باز در ترمینال (سقف ۱)"

        # سقف روزانه
        today = str(now_server.date())
        if self.state.get("last_entry_date") == today and \
                self.state.get("trades_today", 0) >= self.cfg.trading.max_trades_per_day:
            return "سقف معاملات روزانه"

        # سشن (UTC) — ورود در open کندل بعدی
        if not session_open(pd.Series([entry_utc]))[0]:
            return f"خارج از سشن ورود ({entry_utc:%H:%M} UTC)"

        # اسپرد لحظه‌ای
        try:
            q = self.provider.live_quote()
            spread_usd = float(q["spread_points"]) * float(q["point"])
        except Exception as e:  # noqa: BLE001
            return f"تیک ناموجود ({e})"
        if spread_usd > self.profile.max_spread_usd:
            return (f"اسپرد ${spread_usd:.3f} > گیت "
                    f"${self.profile.max_spread_usd} ({self.symbol})")

        # بریکر
        if not self.policy.allow_entry(now_server.to_pydatetime()):
            return "بریکر اجازهٔ ورود نمی‌دهد"

        # رژیم + سیگنال
        reg = RegimeEngine().compute(bars)
        regime_now = int(reg["regime"].iloc[-1])
        inner = V0Strategy(h4, rr=2.5, sl_pad=self.profile.sl_pad)
        inner.prepare(bars)
        sig = inner.on_bar(len(bars) - 1)
        if sig is None:
            return f"بدون سیگنال (رژیم: {REGIME_NAMES.get(regime_now)})"
        if self.direction is not None and sig.direction != self.direction:
            return f"سیگنال {sig.direction:+d} خلاف فیلتر جهت"
        needs = TREND_UP if sig.direction > 0 else TREND_DOWN
        if regime_now != needs:
            return f"رژیم {REGIME_NAMES.get(regime_now)} ≠ {REGIME_NAMES[needs]}"
        if regime_now == CHAOS:
            return "رژیم CHAOS"

        # سایز — ضریب قرارداد از پروفایل نماد (طلا ۱۰۰، نقره ۵۰۰۰)
        quote_bid = q["bid"]
        entry_est = q["ask"] if sig.direction > 0 else quote_bid
        dist = abs(entry_est - sig.stop)
        if dist <= 0:
            return "فاصلهٔ استاپ نامعتبر"
        mult = self.policy.size_multiplier()
        risk_usd = equity * self.cfg.risk.risk_per_trade * mult
        lots = int(risk_usd / (dist * self.profile.contract_size)
                   / 0.01) * 0.01  # گام 0.01
        if lots < 0.01:
            # سیاست حداقل‌لات (تصمیم فاز ۷): طلا force (رفتار قبل)، نقره skip
            if self.profile.min_lot_policy == "skip":
                return (f"حجم {risk_usd / (dist * self.profile.contract_size):.3f} "
                        f"لات < حداقل ۰.۰۱ — استاپِ دوردست، ریسک > بودجه → رد")
            lots = 0.01
        lots = min(lots, self.profile.max_lots)  # سقف عقل‌سنجی دمو
        tp = (entry_est + 2.5 * (entry_est - sig.stop) if sig.direction > 0
              else entry_est - 2.5 * (sig.stop - entry_est))

        order = Order(symbol=self.symbol, direction=sig.direction,
                      units=lots, price=float(entry_est),
                      stop=float(sig.stop), target=float(tp),
                      ts=datetime.now(),
                      meta={"regime": regime_now, "spread_usd": spread_usd})
        if self.dry_run:
            self.paper = PaperPosition(
                id="paper", direction=sig.direction, units=lots,
                entry=float(entry_est), stop=float(sig.stop),
                target=float(tp),
                risk_usd=dist * lots * self.profile.contract_size)
            fill = Fill(order_id="paper", ts=datetime.now(),
                        price=float(entry_est), units=lots)
        else:
            fill = self.adapter.place_order(order)

        risk_usd_final = dist * lots * self.profile.contract_size
        tid = self.journal.open_trade(
            opened_at=datetime.now(), symbol=self.symbol,
            direction=sig.direction, strategy="v0_fvg_gated", grade="A",
            entry=fill.price, stop=float(sig.stop), target=float(tp),
            size_units=lots, regime=REGIME_NAMES.get(regime_now),
            features={"spread_usd": round(spread_usd, 3),
                      "regime": REGIME_NAMES.get(regime_now),
                      "mult": mult, "risk_usd": round(risk_usd_final, 2),
                      "session_utc": f"{entry_utc:%H:%M}",
                      "dry_run": self.dry_run},
            reason="M15 FVG + regime gate")
        self.state["ticket"] = fill.order_id
        self.state["trade_id"] = tid
        self.state["risk_usd"] = risk_usd_final
        tday = str(now_server.date())
        if self.state.get("last_entry_date") != tday:
            self.state["last_entry_date"] = tday
            self.state["trades_today"] = 0
        self.state["trades_today"] = int(self.state.get("trades_today", 0)) + 1
        d = self.profile.digits
        self._print(f"{'🧪' if self.dry_run else '📤'} ورود "
                    f"{'خرید' if sig.direction > 0 else 'فروش'} {lots} لات @ "
                    f"{fill.price:.{d}f} | SL {sig.stop:.{d}f} | TP {tp:.{d}f} | "
                    f"رژیم {REGIME_NAMES.get(regime_now)} | اسپرد ${spread_usd:.3f} "
                    f"| {self.symbol}")
        return "ورود انجام شد"
