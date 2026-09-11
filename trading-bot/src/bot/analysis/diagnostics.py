"""تشخیص خودکار بعد از رشتهٔ باخت — فاز ۴، لایهٔ «مشاهده».

اصل طراحی (درس کالبدشکافی v0): ربات بعد از باخت‌ها حق ندارد
استراتژی‌اش را «عوض کند» — فقط حق دارد «ببیند و ثبت کند».
    - با نرخ برد ۳۳٪، رشتهٔ ۳-۶ باخت ریاضیاتِ نرمال بازی است، نه سیگنالِ
      «استراتژی خراب شده». تغییر قواعد روی همین نویز = کشتن لبه.
    - پس خروجی این ماژول «گزارش» است، نه دستور. هیچ تابعی اینجا
      رفتار معامله‌گری را تغییر نمی‌دهد — به عمد و با تست.

معیارهای تشخیص (همه تفسیرپذیر، هیچ مدل سیاه‌پوستی در کار نیست):
    ۱. آماری: P(این رشته | نرخ‌برد پایه) چقدر است؟
    ۲. رژیم: باخت‌ها در چه رژیمی جمع شده‌اند؟ وسط راه بازار عوض شده؟
    ۳. اسپرد: اسپرد لحظهٔ ورودهای این رشته نابه‌جا بالا بوده؟
    ۴. خوشهٔ زمانی: همه در یک سشن/یک روز، یا پخش؟
    ۵. پوسیدگی لبه: نرخ‌برد ۳۰ معاملهٔ اخیر به‌طور معنادار زیر پایه؟

حکم‌ها: NORMAL_FLUCTUATION | CHAOS_EXPOSURE | REGIME_SHIFT |
         SPREAD_PROBLEM | EDGE_DECAY | INSUFFICIENT_DATA
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

VERDICT_FA = {
    "NORMAL_FLUCTUATION": "نوسان نرمال — اقدامی لازم نیست",
    "CHAOS_EXPOSURE": "مواجهه با آشفتگی بازار",
    "REGIME_SHIFT": "تغییر رژیم بازار",
    "SPREAD_PROBLEM": "مشکل اسپرد/اجرا",
    "EDGE_DECAY": "نشانهٔ ضعف لبه — بازبینی آماری",
    "INSUFFICIENT_DATA": "داده کافی نیست — فقط ثبت",
}


@dataclass
class DiagnosticReport:
    ts: datetime
    streak: int
    verdict: str
    confidence: str                     # low | medium | high
    findings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# ابزار آماری — بدون وابستگی؛ دو جمله‌ای دقیق
# ------------------------------------------------------------------ #
def p_loss_streak(streak: int, win_rate: float) -> float:
    """احتمال رشتهٔ k-باختی از نقطهٔ شروع، با نرخ‌برد پایه."""
    q = 1.0 - win_rate
    return q ** streak


def p_wins_at_most(k: int, n: int, p: float) -> float:
    """P(X ≤ k بردها در n معامله | نرخ‌برد پایه p) — آزمون دقیق دو جمله‌ای."""
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))
    total = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                for i in range(0, k + 1))
    return total


# ------------------------------------------------------------------ #
# تشخیص رشتهٔ باخت
# ------------------------------------------------------------------ #
def diagnose_streak(trades: pd.DataFrame, streak: int,
                    regime_df: pd.DataFrame | None = None,
                    baseline_wr: float | None = None,
                    ts: datetime | None = None) -> DiagnosticReport:
    """trades: معاملاتِ بسته‌شده تا این لحظه (شامل رشتهٔ اخیر در انتها).

    baseline_wr: نرخ‌برد بلندمدت استراتژی (از walk-forward، نه حدس).
    regime_df: DataFrame با time/regime برای چسباندن رژیم لحظهٔ ورود.
    """
    ts = ts or (pd.Timestamp(trades["exit_time"].iloc[-1]) if len(trades)
                else None)
    rep = DiagnosticReport(ts=ts, streak=streak, verdict="INSUFFICIENT_DATA",
                           confidence="low")

    if len(trades) < 30:
        rep.findings.append(
            f"فقط {len(trades)} معامله ثبت شده — حداقل آماری ۳۰ تا است؛ فقط ثبت می‌کنیم.")
        return rep

    losses = trades[trades["pnl"] <= 0]
    wins = trades[trades["pnl"] > 0]
    wr = len(wins) / len(trades)
    base = baseline_wr if baseline_wr is not None else wr

    # ---- ۱) بررسی آماری رشته --------------------------------------
    p_streak = p_loss_streak(streak, base)
    rep.stats["p_streak"] = p_streak
    rep.findings.append(
        f"رشتهٔ {streak}-باختی با نرخ‌برد پایه {base:.0%}: "
        f"احتمال از هر نقطهٔ شروع = {p_streak:.1%}"
        + (" (نرمال)" if p_streak >= 0.05 else " (کم‌احتمال — دقت بیشتر)"))

    # ---- ۲) خوشهٔ زمانی -------------------------------------------
    recent = trades.tail(streak)
    span_h = (pd.Timestamp(recent["exit_time"].iloc[-1])
              - pd.Timestamp(recent["entry_time"].iloc[0])).total_seconds() / 3600
    rep.stats["streak_span_hours"] = span_h
    if span_h < 24:
        rep.findings.append(f"کل رشته در {span_h:.0f} ساعت — خوشهٔ یک سشن/رویداد واحد.")

    # ---- ۳) رژیم لحظهٔ ورودها ---------------------------------------
    if regime_df is not None and not regime_df.empty:
        rg = regime_df[["time", "regime"]].reset_index(drop=True)
        s = pd.Series(rg["regime"].to_numpy(),
                      index=pd.DatetimeIndex(rg["time"]))
        entry_regimes = s.reindex(pd.DatetimeIndex(recent["entry_time"]),
                                  method="ffill")   # آخرین رژیمِ قبل از ورود
        dist = entry_regimes.value_counts(normalize=True)
        rep.stats["regime_dist"] = {str(k): round(float(v), 2)
                                    for k, v in dist.items()}
        if (entry_regimes == 2).mean() >= 0.5:
            rep.verdict = "CHAOS_EXPOSURE"
            rep.confidence = "high"
            rep.findings.append("بیشتر ورودهای این رشته در رژیم CHAOS بوده‌اند.")
            return rep
        # تغییر رژیم بازار در طول رشته: آخرین رژیم ثبت‌شده
        last_reg = rg["regime"].iloc[-1]
        mode = entry_regimes.mode()
        if len(mode) and pd.notna(mode.iloc[0]) and last_reg != mode.iloc[0]:
            rep.verdict = "REGIME_SHIFT"
            rep.confidence = "medium"
            rep.findings.append(
                f"رژیم ورودها {mode.iloc[0]} بود ولی الان {last_reg} — "
                "بازار وسط رشته عوض شده.")
            return rep

    # ---- ۴) اسپرد لحظهٔ ورود ----------------------------------------
    if "spread_usd" in recent.columns and "spread_usd" in trades.columns:
        rs = recent["spread_usd"].mean()
        as_ = trades["spread_usd"].mean()
        rep.stats["streak_spread"] = float(rs)
        rep.stats["avg_spread"] = float(as_)
        if as_ > 0 and rs > 1.5 * as_:
            rep.verdict = "SPREAD_PROBLEM"
            rep.confidence = "medium"
            rep.findings.append(
                f"اسپرد ورودهای رشته ${rs:.2f} در برابر میانگین ${as_:.2f} — "
                "محیط اجرا خراب بوده.")
            return rep

    # ---- ۵) پوسیدگی لبه: ۳۰ معاملهٔ اخیر در برابر پایه ----------------
    window = trades.tail(30)
    w_wins = int((window["pnl"] > 0).sum())
    p_val = p_wins_at_most(w_wins, len(window), base)
    rep.stats["window_wr"] = w_wins / len(window)
    rep.stats["p_window"] = p_val
    if p_val < 0.05:
        rep.verdict = "EDGE_DECAY"
        rep.confidence = "medium"
        rep.findings.append(
            f"نرخ‌برد ۳۰ معاملهٔ اخیر {w_wins/len(window):.0%} در برابر پایه "
            f"{base:.0%} (p={p_val:.3f}) — این دیگر نویز نیست؛ بازبینی آماری لازم است.")
        return rep

    # ---- ۶) پیش‌فرض: نوسان نرمال ------------------------------------
    rep.verdict = "NORMAL_FLUCTUATION"
    rep.confidence = "high" if p_streak >= 0.05 else "low"
    rep.findings.append(
        "هیچ نشانهٔ سیستمیکی پیدا نشد — این همان باخت‌هایی است که سیستم "
        "سودده هم به‌طور روتین تجربه می‌کند. اقدام: هیچ (بریکر سایز را مدیریت می‌کند).")
    return rep
