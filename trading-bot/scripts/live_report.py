"""گزارش فارسی از دموی فوروارد — این را اجرا کن و خروجی را در چت Arena بفرست.

اجرا:
    py scripts/live_report.py
    py scripts/live_report.py --days 7     # فقط ۷ روز اخیر

فاز ۷: چند-نمادی — هر آستین (طلا/نقره) ضربان، معاملات و بریکر خودش را دارد؛
ژورنال مشترک است و ستون symbol جدا می‌کند.
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.postmortem import postmortem
from bot.journal.store import Journal  # فقط برای مهاجرت ستون symbol روی DB قدیمی

# پایهٔ نرخ‌برد پست‌مورتم، به ازای هر نماد (شواهد walk-forward / first-look)
BASELINE_WR = {"XAUUSD": 0.348, "XAGUSD": 0.308}


def main() -> None:
    ap = argparse.ArgumentParser(description="گزارش دمو")
    ap.add_argument("--db", default="data/journal.db")
    ap.add_argument("--days", type=int, default=0, help="فقط N روز اخیر")
    args = ap.parse_args()

    p = pathlib.Path(args.db)
    if not p.exists():
        raise SystemExit(f"❌ ژورنال پیدا نشد: {p} — اول run_live.py را اجرا کن")
    Journal(str(p)).close()   # مهاجرت ستون symbol اگر DB قدیمی است
    conn = sqlite3.connect(p)

    # read_sql_query: ستون‌ها حتی وقتی نتیجه خالی است نام درست دارند
    # (pd.DataFrame(fetchall()) با sqlite3.Row ستون‌های [0,1] می‌سازد — باگ #1)
    def q(sql: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(sql, conn)
        except (sqlite3.OperationalError,
                pd.errors.DatabaseError):   # جدول هنوز ساخته نشده
            return pd.DataFrame()

    trades = q("SELECT * FROM trades WHERE status='closed' ORDER BY closed_at")
    opens = q("SELECT * FROM trades WHERE status='open'")
    eq = q("SELECT * FROM equity ORDER BY ts")
    brk = q("SELECT * FROM breaker_events ORDER BY ts")

    print("═══ گزارش دموی فوروارد (چند-نمادی) ═══")
    if eq.empty:
        print("هنوز داده‌ای ثبت نشده.")
        return

    # ---- ضربان: هر آستین جدا (ردیف‌های قدیمی بدون symbol = طلا) ----
    eq = eq.copy()
    eq["sym"] = eq["symbol"].fillna("XAUUSD") if "symbol" in eq else "XAUUSD"
    eq["ts"] = pd.to_datetime(eq["ts"])
    print("── ضربان ──")
    for sym, g in eq.groupby("sym"):
        age_h = (pd.Timestamp.now() - g["ts"].iloc[-1]).total_seconds() / 3600.0
        last24 = int((g["ts"] >= pd.Timestamp.now()
                      - pd.Timedelta(hours=24)).sum())
        mark = "✅" if age_h <= 2.0 else "⚠️"
        print(f"{mark} {sym}: آخرین چرخه {g['ts'].iloc[-1]:%m-%d %H:%M} "
              f"({age_h:.1f} ساعت پیش) | ۲۴ساعت اخیر: {last24} | کل: {len(g)}")
    if not trades.empty:
        trades["closed_at"] = pd.to_datetime(trades["closed_at"])
        trades["opened_at"] = pd.to_datetime(trades["opened_at"])
        if args.days:
            cut = pd.Timestamp.now() - pd.Timedelta(days=args.days)
            trades = trades[trades["closed_at"] >= cut]

    open_syms = sorted(opens["symbol"].unique()) if not opens.empty else []
    print(f"معاملات بسته‌شده: {len(trades)} | باز: {len(opens)}"
          + (f" ({', '.join(open_syms)})" if open_syms else ""))

    # ---- هر آستین جداگانه ----
    for sym, t in (trades.groupby("symbol") if not trades.empty else []):
        base = BASELINE_WR.get(sym, 0.34)
        print(f"\n───── آستین {sym} (پایهٔ برد {base:.1%}) ─────")
        wins = (t["r_multiple"] > 0).sum()
        print(f"نرخ برد: {wins}/{len(t)} = {wins/len(t):.0%} | "
              f"میانگین R: {t['r_multiple'].mean():.3f} | "
              f"جمع R: {t['r_multiple'].sum():.2f}")
        print("آخرین معاملات:")
        cols = ["opened_at", "direction", "entry", "exit_price",
                "r_multiple", "regime"]
        show = t[cols].tail(10).copy()
        show["direction"] = show["direction"].map({1: "خرید", -1: "فروش"})
        show["opened_at"] = show["opened_at"].dt.strftime("%m-%d %H:%M")
        print(show.to_string(index=False,
                             float_format=lambda v: f"{v:,.2f}"))

        if len(t) >= 10:
            print(f"\n── پست‌مورتم {sym} ──")
            tt = t.rename(columns={"opened_at": "entry_time",
                                   "closed_at": "exit_time",
                                   "r_multiple": "pnl"})
            pm = postmortem(tt, baseline_wr=base, min_trades=10)
            print(f"توصیه: {pm['recommendation']} — {pm['reason']}")
            g = tt.groupby("direction")["pnl"].agg(["count", "sum"]).round(2)
            g.index = g.index.map({1: "خرید", -1: "فروش"})
            print("جهت (بر حسب R):"); print(g.to_string())
            if "regime" in tt.columns:
                g2 = tt.groupby("regime")["pnl"].agg(["count", "sum"]).round(2)
                print("رژیم ورود (بر حسب R):"); print(g2.to_string())

    if not brk.empty:
        print(f"\n── رویدادهای بریکر ({len(brk)}) ──")
        if "symbol" in brk:
            print(brk.groupby(["symbol", "kind"], dropna=False).size()
                  .to_string())
        else:
            print(brk.groupby("kind").size().to_string())
        recent = brk.tail(5)
        for _, r in recent.iterrows():
            sym = (r.get("symbol") or "?") if "symbol" in brk else "?"
            print(f"  {r['ts'][:16]} [{sym}] {r['kind']}: {r['detail'][:70]}")

    print("\n── سرمایهٔ هر آستین (مبنای بریکر ماهانه) ──")
    for sym, g in eq.groupby("sym"):
        g = g.sort_values("ts")
        e = g["equity"].astype(float)
        peak = e.cummax()
        dd = ((e - peak) / peak).min()
        print(f"{sym}: آخرین ${e.iloc[-1]:,.0f} | بدترین افت ثبت‌شده {dd:.1%}")
    print("\n(این خروجی را کامل در چت Arena بفرست تا با هم تحلیل کنیم)")


if __name__ == "__main__":
    main()
