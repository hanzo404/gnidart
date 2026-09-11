"""متا-لیبلینگ نسخهٔ ۲ — سایزینگ احتمالاتی (فاز ۶-ب).

فلسفه (درسِ شکست ۶-۱): فیلترِ سخت «همه یا هیچ» در OOS مرد. این‌جا هیچ
معامله‌ای حذف نمی‌شود؛ به‌جایش حجم هر معامله تابع احتمالِ بردِ پیش‌بینی‌شده
است — خطاهای مدل دیگر فاجعه نیستند، فقط کوچک/بزرگ می‌شوند.

طرح از پیش ثبت‌شده (قبل از دیدن هر نتیجه‌ای — ۲۰۲۶-۰۹-۱۱):
    - استخر کاندیدا: دقیقاً همان ۶-۱ (بدون گیت رژیم، سشن روشن، فقط خرید،
      لات ثابت 0.01، بدون بریکر) — مقایسه‌پذیر با نتایج ۶-۱.
    - پنجره‌ها: همان purged_monthly_windows با IS=12 ماه، OOS=1 ماه،
      خالص‌سازی خروج‌های IS داخل OOS.
    - مدل‌ها: logit و GBM (همان ۶-۱)؛ انتخاب مدل در هر پنجره فقط با
      IS log-loss (سنجه‌ی کالیبراسیون — نه گیمینگِ PF).
    - سه تابع سایزینگ، همه با کلامپ [0.5, 1.5] (هیچ معامله‌ای صفر نمی‌شود):
        stepped: p<0.35→0.5 | 0.35≤p<0.45→1.0 | p≥0.45→1.5  (باندهای ثابت)
        linear:  m = clamp(p / p_base, 0.5, 1.5)  (p_base = نرخ برد IS)
        kelly:   m = clamp(f*(p)/f*(p_base), 0.5, 1.5)، f* = p − (1−p)/2.5
    - بازوی شاهد اوراکل (m=1.5 برای برد، 0.5 برای باخت) = سقف تئوریکِ
      وزن‌دهیِ کامل — فقط برای درک فاصله؛ قابل‌معامله نیست (لو-اِهد).
    - میله‌ی موفقیت (MT5، OOS دوازده‌ماههٔ متحرک): PF وزنی ≥ 1.4 با
      فرکانس ≥ ۱۳/ماه؛ و تکرار روی Dukascopy: PF وزنی ≥ 1.3 و بهتر از
      take_all. رد شود → دفن صادقانه، G6 می‌ماند.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.analysis.meta import (FEATURES, MetaWFOConfig, fit_probs,
                               purged_monthly_windows)

M_MIN, M_MAX = 0.5, 1.5
B_RR = 2.5          # نسبت سود به ضرر استراتژی پایه — برای کلی
SIZING_COLS = ("m_stepped", "m_linear", "m_kelly")


# ------------------------------------------------------------------ #
# توابع سایزینگ — همگی برداری، همگی کلامپ‌شده، هیچ‌کدام صفر نمی‌کنند
# ------------------------------------------------------------------ #
def sizing_stepped(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.35, 0.5, np.where(p < 0.45, 1.0, 1.5))


def sizing_linear(p, p_base: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p / max(p_base, 1e-9), M_MIN, M_MAX)


def sizing_kelly(p, p_base: float) -> np.ndarray:
    """کلیِ کسری برای برد/باخت دودویی در R=±(1, B_RR): f* = p − (1−p)/B_RR.

    نرمال‌سازی به میانگین ۱ (تقسیم بر f*(p_base)) تا ریسکِ متوسط تغییر
    نکند و مقایسه با take_all منصفانه باشد. اگر p_base آن‌قدر پایین بود
    که f*(p_base) ≤ 0، به linear برمی‌گردیم (کلی کاربردی ندارد).
    """
    p = np.asarray(p, dtype=float)
    f = p - (1.0 - p) / B_RR
    f0 = p_base - (1.0 - p_base) / B_RR
    if f0 <= 0:
        return sizing_linear(p, p_base)
    return np.clip(f / f0, M_MIN, M_MAX)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------ #
# walk-forward سایزینگ
# ------------------------------------------------------------------ #
@dataclass
class SizingWFOResult:
    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos: pd.DataFrame = field(default_factory=pd.DataFrame)   # + p, m_*


def run_meta_sizing_wfo(ds: pd.DataFrame,
                        cfg: MetaWFOConfig | None = None) -> SizingWFOResult:
    """هر پنجره: فیت روی IS خالص‌شده → انتخاب مدل با IS log-loss →
    احتمال OOS → سه ضریب سایز + اوراکل. خروجی: معامله‌های OOS با m‌ها."""
    cfg = cfg or MetaWFOConfig()
    wins = purged_monthly_windows(ds, cfg.is_months, cfg.oos_months)
    rows, frames = [], []
    for is_idx, oos_idx, label in wins:
        if len(is_idx) < cfg.min_kept_is:
            continue
        y_is = ds["win"].to_numpy()[is_idx]
        p_base = float(y_is.mean())
        probs = fit_probs(ds, is_idx, oos_idx, cfg)

        best_name, best_ll, p_oos_best = None, np.inf, None
        for name, pr in probs.items():
            if pr is None:
                continue
            p_is, p_oos = pr
            ll = _logloss(p_is, y_is)
            if ll < best_ll:
                best_name, best_ll, p_oos_best = name, ll, p_oos
        if best_name is None:                     # هیچ مدلی فیت نشد
            best_name, p_oos_best = "base_rate", np.full(len(oos_idx), p_base)

        oos = ds.iloc[oos_idx].copy()
        oos["p"] = p_oos_best
        oos["model"] = best_name
        oos["p_base"] = p_base
        oos["m_stepped"] = sizing_stepped(oos["p"])
        oos["m_linear"] = sizing_linear(oos["p"], p_base)
        oos["m_kelly"] = sizing_kelly(oos["p"], p_base)
        oos["m_oracle"] = np.where(oos["win"].to_numpy() > 0, 1.5, 0.5)
        frames.append(oos)
        rows.append({"oos_month": label, "is_n": len(is_idx),
                     "p_base": round(p_base, 3), "model": best_name,
                     "is_logloss": round(best_ll, 4)
                     if best_ll < np.inf else None,
                     "oos_n": len(oos), "mean_p": round(float(oos["p"].mean()), 3),
                     "oos_wr": round(float(oos["win"].mean()), 3)})
    per = pd.DataFrame(rows)
    return SizingWFOResult(
        per_window=per,
        oos=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


# ------------------------------------------------------------------ #
# جمع‌بندی وزنی
# ------------------------------------------------------------------ #
def agg_weighted(frame: pd.DataFrame, mcol: str, n_windows: int) -> dict:
    """متریک‌های OOS با ضریب m: PF/PnL وزنی، انتظارِ سرمایه‌وزنی، DD ریسکی."""
    if not len(frame):
        return {"n": 0, "per_month": 0.0, "wr": float("nan"),
                "pf": float("nan"), "pnl": 0.0, "exp_r": float("nan"),
                "mean_m": float("nan"), "dd_r": 0.0}
    m = frame[mcol].to_numpy(dtype=float)
    pnl = frame["pnl"].to_numpy(dtype=float) * m
    r = frame["r"].to_numpy(dtype=float) * m
    gw = float(pnl[pnl > 0].sum())
    gl = -float(pnl[pnl <= 0].sum())
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    dd_r = float((eq - peak).min()) if len(eq) else 0.0
    return {"n": len(frame), "per_month": len(frame) / max(n_windows, 1),
            "wr": float((frame["pnl"] > 0).mean()),
            "pf": gw / gl if gl else float("inf"),
            "pnl": float(pnl.sum()),
            "exp_r": float(r.sum() / m.sum()) if m.sum() else float("nan"),
            "mean_m": float(m.mean()), "dd_r": dd_r}
