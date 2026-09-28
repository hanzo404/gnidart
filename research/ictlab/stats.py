"""آمارِ استراتژی: امید ریاضی، افت سرمایه، معناداری، و تلهٔ واریانس.

نکتهٔ کلیدی که کتاب‌های این کتابخانه بارها تکرار می‌کنند
(و که در تست ۱۰سالهٔ مستقل ICT هم دیده شد):
    نرخ بردِ بالا دلیلِ سودآوری نیست.
این ماژول همیشه «منحنی سرمایه با سایزینگ درصدیِ ثابت» را هم می‌سازد،
چون همان‌جا است که سیستم‌های امید‌ریاضی مثبت می‌میرند.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(trades: pd.DataFrame, risk_pct: float = 0.5, equity0: float = 10_000.0) -> dict:
    if trades is None or trades.empty:
        return {"n": 0}
    r = trades.r.values
    wins = r[r > 0]
    losses = r[r < 0]
    n = len(r)
    gross_win, gross_loss = wins.sum(), -losses.sum()
    eq = equity0 * np.cumprod(1 + r * risk_pct / 100.0)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "n": n,
        "win_rate": 100 * len(wins) / n,
        "expectancy_r": r.mean(),
        "total_r": r.sum(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "avg_win_r": wins.mean() if len(wins) else 0.0,
        "avg_loss_r": losses.mean() if len(losses) else 0.0,
        "payoff": (wins.mean() / -losses.mean()) if len(wins) and len(losses) else 0.0,
        "max_dd_pct": 100 * dd.min(),
        "final_equity": eq[-1],
        "max_consec_loss": int(_max_run(r < 0)),
        "tp_rate": 100 * (trades.reason == "tp").mean(),
        "sl_rate": 100 * (trades.reason == "sl").mean(),
        "avg_bars": trades.bars_held.mean(),
        "avg_mfe": trades.mfe.mean(),
        "avg_mae": trades.mae.mean(),
    }


def _max_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def bootstrap_tstat(r: np.ndarray, n_boot: int = 5000, seed: int = 7) -> float:
    """t آماریِ امید ریاضی با بوت‌استرپ (روش پیشنهادی لُ در Evidence-Based TA)."""
    if len(r) < 20:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    means = r[idx].mean(axis=1)
    return float(means.mean() / (means.std() + 1e-12))


def block_bootstrap_tstat(trades: pd.DataFrame, n_boot: int = 4000, seed: int = 13) -> dict:
    """بوت‌استرپ بلوکی بر پایهٔ «روز».

    معاملاتِ هم‌دُم به‌شدت همبسته‌اند؛ بوت‌استرپ سادهٔ معامله‌به‌معامله
    (که لُ توصیه می‌کند) تعداد مؤثر نمونه را بیش‌برآورد می‌کند. اینجا
    کل روزها با هم بازنمونه‌برداری می‌شوند.
    """
    if trades is None or trades.empty:
        return {"t": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "days": 0}
    t = trades.copy()
    t["day"] = t.time.dt.date
    days = t.groupby("day").r.mean().values      # میانگین روزانه = واحد مستقل
    n = len(days)
    if n < 20:
        return {"t": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "days": n}
    obs = days.mean()
    rng = np.random.default_rng(seed)
    means = days[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    se = means.std()
    return {"t": float(obs / (se + 1e-12)), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "days": int(n),
            "daily_mean_r": float(obs), "daily_sd": float(days.std())}


def monthly_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["month"] = t.time.dt.to_period("M")
    g = t.groupby("month").agg(n=("r", "size"), r_sum=("r", "sum"),
                               pf=("r", lambda s: s[s > 0].sum() / -s[s < 0].sum() if (s < 0).any() else np.inf))
    g["wr"] = t.groupby("month").r.apply(lambda s: 100 * (s > 0).mean())
    return g


def halve_report(trades: pd.DataFrame) -> pd.DataFrame:
    """تقسیم به نصف اول/دوم — آزمون صادقانهٔ پایداری (خط پایهٔ این ریپو)."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.sort_values("time").reset_index(drop=True)
    mid = len(t) // 2
    rows = []
    for label, part in (("H1", t.iloc[:mid]), ("H2", t.iloc[mid:]),
                        ("Y1", t[t.time < t.time.min() + pd.Timedelta(days=365)]),
                        ("Y2+", t[t.time >= t.time.min() + pd.Timedelta(days=365)])):
        m = metrics(part)
        if m.get("n"):
            rows.append({"slice": label, **m})
    return pd.DataFrame(rows)


def equity_curve(trades: pd.DataFrame, risk_pct: float = 0.5, equity0: float = 10_000.0) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    r = trades.sort_values("time").r.values
    return pd.Series(equity0 * np.cumprod(1 + r * risk_pct / 100.0), index=trades.sort_values("time").time.values)
