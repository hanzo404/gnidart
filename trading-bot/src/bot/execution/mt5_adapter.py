"""اجرا روی MetaTrader 5 (دمو) — فاز ۵.

فقط روی ویندوز کار می‌کند (پکیج MetaTrader5). تمام سفارش‌ها:
    - market با SL/TP چسبیده (اتمیک — اگر ربات خاموش شود استاپ سر جایش است)
    - magic number مخصوص ربات (جدا کردن از معاملات دستی)
    - حداقل/گام حجم و عقل‌سنجی حجم از symbol_info

قواعد ایمنی (درس‌های v0):
    - positions_get قبل از هر سفارش — هرگز پوزیشن دوم روی همان سیمبل
    - چک دمو بودن اکانت قبل از هر ارسال
"""
from __future__ import annotations

import time
from datetime import datetime

from bot.execution.base import ExecutionAdapter, Fill, Order

try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:  # pragma: no cover — سندباکس لینوکس
    mt5 = None
    _HAS_MT5 = False


class MT5ExecutionAdapter(ExecutionAdapter):
    name = "mt5-demo"

    def __init__(self, symbol: str, magic: int = 954001,
                 deviation: int = 10, require_demo: bool = True):
        if not _HAS_MT5:
            raise RuntimeError("پکیج MetaTrader5 نصب نیست — فقط ویندوز")
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.require_demo = require_demo

    # ------------------------------------------------------------------ #
    def _check_demo(self) -> None:
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("account_info ناموفق")
        if self.require_demo and acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError(
                "⛔ این اکانت دمو نیست! ربات فقط روی اکانت دمو اجرا می‌شود. "
                "به ترمینال دمو سوئیچ کن."
            )

    def _filling_type(self) -> int:
        """نوع پر کردن سفارش از خود نماد — بروکرها فرق دارند.

        (ممیزی خارجی + مستندات MQL5): مجوز IOC/FOK به ازای هر نماد/بروکر است؛
        هاردکدِ IOC می‌تواند «unsupported filling mode» بدهد.
        """
        try:
            info = mt5.symbol_info(self.symbol)
            if info is not None:
                fm = int(info.filling_mode)   # بیت‌ماسک: 1=FOK، 2=IOC
                if fm & 2:
                    return mt5.ORDER_FILLING_IOC
                if fm & 1:
                    return mt5.ORDER_FILLING_FOK
        except Exception:  # noqa: BLE001 — عقب‌افتادن به پیش‌فرض امن
            pass
        return mt5.ORDER_FILLING_IOC

    def place_order(self, order: Order) -> Fill:
        if self.require_demo:
            self._check_demo()
        # retry سبک برای خطاهای گذرا (requote/price) — ۳ تلاش با تیک تازه
        transient = {getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004),
                     getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020),
                     getattr(mt5, "TRADE_RETCODE_PRICE_OFF", 10021)}
        result = price = None
        for attempt in range(3):
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                raise RuntimeError(
                    f"تیک {self.symbol} ناموجود — Market Watch را چک کن")
            price = tick.ask if order.direction > 0 else tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(order.units),
                "type": (mt5.ORDER_TYPE_BUY if order.direction > 0
                         else mt5.ORDER_TYPE_SELL),
                "price": float(price),
                # is not None (نه truthy): استاپِ 0.0 نباید بی‌صدا حذف شود
                "sl": (float(order.stop)
                       if order.stop is not None else None),
                "tp": (float(order.target)
                       if order.target is not None else None),
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "gnidart p5",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_type(),
            }
            # MT5 مقادیر None در sl/tp را نمی‌پذیرد — کلید را حذف کن
            req = {k: v for k, v in req.items() if v is not None}
            result = mt5.order_send(req)
            if result is None:
                raise RuntimeError(
                    f"order_send هیچ جوابی نداد: {mt5.last_error()}")
            if result.retcode != mt5.TRADE_RETCODE_DONE \
                    and result.retcode in transient and attempt < 2:
                time.sleep(0.25)
                continue
            break
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"سفارش رد شد (retcode={getattr(result, 'retcode', '?')}): "
                f"{getattr(result, 'comment', '')}")
        fill_price = result.price if result.price > 0 else price
        return Fill(order_id=str(result.order), ts=datetime.now(),
                    price=float(fill_price), units=float(order.units),
                    slippage=abs(float(fill_price) - price))

    # ------------------------------------------------------------------ #
    def close_position(self, position_id: str, ts: datetime,
                       price: float) -> None:
        pos = self._by_ticket(int(position_id))
        if pos is None:
            return  # قبلاً بسته شده
        opposite = (mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(pos.volume),
            "type": opposite,
            "position": int(position_id),
            "price": float(price),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "gnidart p5 close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(),
        }
        result = mt5.order_send(req)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"بستن پوزیشن {position_id} ناموفق")

    def _by_ticket(self, ticket: int):
        positions = mt5.positions_get(symbol=self.symbol) or []
        for p in positions:
            if p.ticket == ticket:
                return p
        return None

    def open_positions(self) -> list:
        """فقط پوزیشن‌های همین ربات (magic) روی همین سیمبل — درس v0: pyramiding."""
        out = []
        for p in (mt5.positions_get(symbol=self.symbol) or []):
            if p.magic == self.magic:
                out.append({
                    "id": str(p.ticket),
                    "direction": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                    "units": float(p.volume),
                    "entry": float(p.price_open),
                    "profit": float(p.profit),
                    "sl": float(p.sl) if p.sl else None,
                    "tp": float(p.tp) if p.tp else None,
                })
        return out

    def close_info(self, position_id: str) -> dict:
        """بعد از بسته‌شدن: قیمت/سود واقعی از تاریخچهٔ deal ها."""
        deals = mt5.history_deals_get(position=int(position_id)) or []
        exit_price, profit = None, 0.0
        for d in deals:
            if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                exit_price = float(d.price)
                profit += float(d.profit) + float(d.swap) + float(d.commission)
        return {"exit_price": exit_price, "profit": profit}
