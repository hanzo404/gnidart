"""گزارش فارسی از دموی فوروارد — این را اجرا کن و خروجی را در چت Arena بفرست.

اجرا:
    py scripts/live_report.py
    py scripts/live_report.py --days 7     # فقط ۷ روز اخیر
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bot.analysis.postmortem import postmortem


def main() -> None:
    ap = argparse.ArgumentParser(description="گزارش دمو")
    ap.add_argument("--db", default="data/journal.db")
    ap.add_argument("--days", type=int, default=0, help="فقط N روز اخیر")
    args = ap.parse_args()

    p = pathlib.Path(args.db)
    if not p.exists():
        raise SystemExit(f"❌ ژورنال پیدا نشد: {p} — اول run_live.py را اجرا کن")
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row

    trades = pd.DataFrame(conn.execute(
        "SELECT * FROM trades WHERE status='closed' ORDER BY closed_at"
    ).fetchall())
    opens = pd.DataFrame(conn.execute(
        "SELECT * FROM trades WHERE status='open'").fetchall())
    eq = pd.DataFrame(conn.execute(
        "SELECT * FROM equity ORDER BY ts").fetchall())
    brk = pd.DataFrame(conn.execute(
        "SELECT * FROM breaker_events ORDER BY ts").fetchall())

    print("═══ گزارش دموی فوروارد ═══")
    if eq.empty:
        print("هنوز داده‌ای ثبت نشده.")
        return
    if not trades.empty:
        trades["closed_at"] = pd.to_datetime(trades["closed_at"])
        trades["opened_at"] = pd.to_datetime(trades["opened_at"])
        if args.days:
            cut = pd.Timestamp.now() - pd.Timedelta(days=args.days)
            trades = trades[trades["closed_at"] >= cut]
    print(f"معاملات بسته‌شده: {len(trades)} | باز: {len(opens)}")

    if not trades.empty:
        wins = (trades["r_multiple"] > 0).sum()
        print(f"نرخ برد: {wins}/{len(trades)} = {wins/len(trades):.0%} | "
              f"میانگین R: {trades['r_multiple'].mean():.3f} | "
              f"جمع R: {trades['r_multiple'].sum():.2f}")
        print("\nآخرین معاملات:")
        cols = ["opened_at", "direction", "entry", "exit_price", "r_multiple",
                "regime"]
        show = trades[cols].tail(10).copy()
        show["direction"] = show["direction"].map({1: "خرید", -1: "فروش"})
        show["opened_at"] = show["opened_at"].dt.strftime("%m-%d %H:%M")
        print(show.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        if len(trades) >= 10:
            print("\n── پست‌مورتم (پایه: ۳۴.۸٪ از walk-forward) ──")
            t = trades.rename(columns={"opened_at": "entry_time",
                                       "closed_at": "exit_time",
                                       "r_multiple": "pnl"})
            pm = postmortem(t, baseline_wr=0.348, min_trades=10)
            print(f"توصیه: {pm['recommendation']} — {pm['reason']}")
            g = t.groupby("direction")["pnl"].agg(["count", "sum"]).round(2)
            g.index = g.index.map({1: "خرید", -1: "فروش"})
            print("جهت (بر حسب R):"); print(g.to_string())
            if "regime" in t.columns:
                g2 = t.groupby("regime")["pnl"].agg(["count", "sum"]).round(2)
                print("رژیم ورود (بر حسب R):"); print(g2.to_string())

    if not brk.empty:
        print(f"\n── رویدادهای بریکر ({len(brk)}) ──")
        print(brk.groupby("kind").size().to_string())
        recent = brk.tail(5)
        for _, r in recent.iterrows():
            print(f"  {r['ts'][:16]} {r['kind']}: {r['detail'][:70]}")

    if not eq.empty:
        eq["equity"] = eq["equity"].astype(float)
        peak = eq["equity"].cummax()
        dd = ((eq["equity"] - peak) / peak).min()
        print(f"\nequity آخرین: ${eq['equity'].iloc[-1]:,.0f} | "
              f"بدترین افت ثبت‌شده: {dd:.1%}")
    print("\n(این خروجی را کامل در چت Arena بفرست تا با هم تحلیل کنیم)")


if __name__ == "__main__":
    main()
