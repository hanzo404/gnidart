"""پست‌مورتم آماری — فاز ۴: تحلیل دوره‌ای روی نمونهٔ بزرگ (نه ۲ معامله!).

تفاوت با diagnostics.py: آن یکی «لحظهٔ رشتهٔ باخت» را نگاه می‌کند؛
این یکی هر ماه / هر ۵۰ معامله روی «کل نمونه» قضاوت می‌کند — همان چیزی
که از روز اول توافق کردیم: تغییر قضاوت فقط روی ۲۰+ معامله، هرگز روی ۲ تا.

خروجی: جدول شکستگی (رژیم/جهت/ساعت) + توصیهٔ مستند:
    CONTINUE      | شواهد کافی برای تغییر نیست — ادامهٔ عادی
    DERATE        | افت معنادار ولی قابل‌مدیریت — نردبان بریکر کافی است
    HALT_REVIEW   | لبه معناداراً زیر آب — توقف تا بازبینی walk-forward انسانی
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.analysis.diagnostics import p_wins_at_most


def postmortem(trades: pd.DataFrame, baseline_wr: float,
               min_trades: int = 30) -> dict:
    """قضاوت آماری روی کل نمونهٔ معاملات + شکستگی‌های تشخیصی."""
    out: dict = {"n_trades": len(trades)}
    if len(trades) < min_trades:
        out["recommendation"] = "CONTINUE"
        out["reason"] = (f"نمونه {len(trades)} < {min_trades} — "
                         "قضاوت آماری معنا ندارد؛ فقط ثبت.")
        return out

    wins = int((trades["pnl"] > 0).sum())
    n = len(trades)
    wr = wins / n
    p = p_wins_at_most(wins, n, baseline_wr)
    gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_l = -trades.loc[trades["pnl"] <= 0, "pnl"].sum()
    pf = float(gross_w / gross_l) if gross_l > 0 else float("inf")

    out.update({"win_rate": wr, "p_value": p, "profit_factor": pf,
                "expectancy": float(trades["pnl"].mean()),
                "expectancy_r": float(trades["r"].mean())})

    if p < 0.01 and wr < baseline_wr * 0.7:
        out["recommendation"] = "HALT_REVIEW"
        out["reason"] = (f"نرخ‌برد {wr:.0%} به‌طور معنادار زیر پایه "
                         f"{baseline_wr:.0%} (p={p:.4f}) — توقف تا بازبینی انسانی.")
    elif p < 0.05 or pf < 1.0:
        out["recommendation"] = "DERATE"
        out["reason"] = (f"افت معنادار (p={p:.3f}, PF={pf:.2f}) — نردبان بریکر "
                         "فعال بماند؛ بازبینی در چک‌پوینت بعدی.")
    else:
        out["recommendation"] = "CONTINUE"
        out["reason"] = (f"نرخ‌برد {wr:.0%} با پایه {baseline_wr:.0%} سازگار "
                         f"(p={p:.2f}) — نوسان عادی نمونه.")

    # ---- شکستگی‌های تشخیصی (کجا می‌سازیم، کجا می‌بازیم) ----
    out["by_direction"] = trades.groupby(
        trades["direction"].map({1: "خرید", -1: "فروش"}))["pnl"] \
        .agg(["count", "sum"]).round(1).to_dict("index")
    out["by_reason"] = trades.groupby("reason")["pnl"] \
        .agg(["count", "sum"]).round(1).to_dict("index")
    hrs = pd.to_datetime(trades["entry_time"]).dt.hour
    out["by_session_half"] = trades.assign(half=np.where(
        (hrs >= 12) & (hrs < 16), "۱۲–۱۶ (لندن+نیویورک)",
        np.where((hrs >= 16) & (hrs < 20), "۱۶–۲۰ (نیویورک)", "سایر"))) \
        .groupby("half")["pnl"].agg(["count", "sum"]).round(1).to_dict("index")
    return out


def attach_regime(trades: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """ستون رژیمِ لحظهٔ ورود را به معاملات می‌چسباند (ffill = آخرین رژیم دیده‌شده)."""
    out = trades.copy()
    if not len(regime_df):
        return out
    s = pd.Series(regime_df["regime"].to_numpy(),
                  index=pd.DatetimeIndex(regime_df["time"]))
    out["entry_regime"] = s.reindex(
        pd.DatetimeIndex(out["entry_time"]), method="ffill").to_numpy()
    return out
