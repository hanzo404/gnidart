"""متا-لیبلینگ — فاز ۶: «این ستاپِ مشخص خوب از آب درمی‌آید یا نه؟»

ایده (López de Prado):
    استراتژی اولیه کاندیدا می‌سازد (اینجا: گیت‌شدهٔ شل — بدون گیت رژیم،
    فقط سشن + فقط خرید) و یک مدلِ یادگیری ماشین در «لحظه‌ی ورود» احتمال
    موفقیت هر کاندیدا را می‌گوید. معامله فقط وقتی انجام می‌شود که
    احتمال از آستانه بگذرد.

درس‌های بازبینی ریپوها (۲۰۲۶-۰۹) که اینجا اعمال شده:
    - فیچرها فقط از «گذشتهٔ» کندل سیگنال (ضد-نشتی؛ تست علیّت دارد)
    - لیبل = نتیجهٔ واقعی موتور بک‌تست (موانع روی H/L + اسپرد + اسلیپیج)
      — نه close-only مثل ریپوی CNN+LSTM
    - scaler و مدل فقط داخل پنجرهٔ IS فیت می‌شوند (ضد نشتیِ scaler)
    - purge: معامله‌های IS که خروجشان به OOS می‌خورد از آموزش حذف می‌شوند
    - مدل‌ها ساده‌اند (LogisticRegression / GradientBoosting کوچک) —
      با ~۵۰۰ نمونه، شبکه‌ی عمیق فقط حفظ می‌کند
    - «همه را بگیر» هم یک گزینه در گرید انتخاب است — اگر فیلتر کمک
      نکند، خودش را کنار می‌گذارد؛ نه با نظر، با IS-pnl
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bot.analysis.walkforward import monthly_windows

# ------------------------------------------------------------------ #
# ۱) اندیکاتورهای علّی (فقط گذشته)
# ------------------------------------------------------------------ #
def compute_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """اندیکاتورها روی کل کندل‌ها — همه فقط از گذشته محاسبه می‌شوند."""
    o = bars["open"].astype(float)
    h = bars["high"].astype(float)
    l = bars["low"].astype(float)
    c = bars["close"].astype(float)

    out = pd.DataFrame(index=bars.index)
    # نسبت‌های کندل (فیچرهای ریپوی CNN — سبک اما مفید)
    total = (h - l).replace(0, np.nan)
    out["body_ratio"] = (c - o) / total
    out["upper_shadow_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / total
    out["lower_shadow_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / total

    # EMA / RSI(Wilder) / ATR(Wilder)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = gain.ewm(alpha=1 / 14, adjust=False).mean() / \
        loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr"] = atr

    # فاصله‌های نرمال با ATR
    out["dist_ema20"] = (c - ema20) / atr
    out["dist_ema50"] = (c - ema50) / atr
    out["dist_ema200"] = (c - ema200) / atr

    # بولینگر 20
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    out["bb_hband_dist"] = (mid + 2 * sd - c) / atr
    out["bb_lband_dist"] = (c - (mid - 2 * sd)) / atr
    out["bb_width_atr"] = (4 * sd) / atr

    # FVG صعودی سه‌کندلی (همان تریگر استراتژی)
    out["fvg_gap_atr"] = (l - h.shift(2)) / atr
    return out


# ------------------------------------------------------------------ #
# ۲) ساخت دیتاست: یک سطر به ازای هر کاندیدا
# ------------------------------------------------------------------ #
FEATURES: list[str] = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
    "rsi14", "atr_close", "dist_ema20", "dist_ema50", "dist_ema200",
    "bb_hband_dist", "bb_lband_dist", "bb_width_atr", "fvg_gap_atr",
    "regime_up", "regime_range", "regime_down", "regime_chaos",
    "bias", "stop_dist_atr", "spread_usd", "session_frac",
    "hours_since_prev", "roll10_wr", "roll10_r",
]


def build_dataset(bars: pd.DataFrame, h4: pd.DataFrame, regime: np.ndarray,
                  sess: np.ndarray, trades: pd.DataFrame) -> pd.DataFrame:
    """trades (خروجی Backtester با لات ثابت) + فیچرهای لحظه‌ی تصمیم.

    نکته‌ی ضد-نشتی: کندلِ سیگنال = entry_i − 1 (ورود در open بعدی).
    همه‌ی فیچرها تا پایان همان کندل محاسبه می‌شوند — دقیقاً چیزی که
    در لایو در لحظه‌ی تصمیم می‌دانیم.
    """
    ind = compute_indicators(bars)
    t = pd.to_datetime(bars["time"])
    times = t.to_numpy()
    reg = np.asarray(regime)

    # بایاس H4 — همان مسیر کد استراتژی (بدون duplicate منطق)
    from bot.backtest.v0_strategy import V0Strategy
    vs = V0Strategy(h4, rr=2.5)
    vs.prepare(bars)
    bias = np.asarray(vs._bias)          # noqa: SLF001 — عمداً همان محاسبه

    tr = trades.copy()
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr = tr.sort_values("entry_time").reset_index(drop=True)

    rows = []
    for k, r in tr.iterrows():
        # ایندکس کندل ورود → کندل سیگنال یکی قبلش
        entry_i = int(np.searchsorted(times, np.datetime64(r["entry_time"])))
        if entry_i <= 2 or entry_i >= len(bars):
            continue
        sig_i = entry_i - 1
        et = r["entry_time"]
        hour = et.hour + et.minute / 60.0
        dow = et.dayofweek
        a = float(ind["atr"].iloc[sig_i])
        if not np.isfinite(a) or a <= 0:
            continue
        # اسپردِ «لحظه‌ی تصمیم» = کندل سیگنال (اسپرد کندل ورود هنوز نامعلوم است)
        spread = (float(bars["spread_usd"].iloc[sig_i])
                  if "spread_usd" in bars.columns else 0.0)
        rv = int(reg[sig_i])
        rows.append({
            # هویت + لیبل
            "entry_time": r["entry_time"], "exit_time": r["exit_time"],
            "pnl": float(r["pnl"]), "r": float(r["r"]),
            "win": int(float(r["pnl"]) > 0),
            # زمان چرخه‌ای
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_sin": np.sin(2 * np.pi * dow / 7.0),
            "day_cos": np.cos(2 * np.pi * dow / 7.0),
            # فیچرهای کندل سیگنال
            **{f: float(ind[f].iloc[sig_i]) for f in (
                "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
                "rsi14", "dist_ema20", "dist_ema50", "dist_ema200",
                "bb_hband_dist", "bb_lband_dist", "bb_width_atr",
                "fvg_gap_atr")},
            "atr_close": a / float(bars["close"].iloc[sig_i]),
            "stop_dist_atr": (float(bars["close"].iloc[sig_i])
                              - float(r["stop"])) / a,
            "spread_usd": spread,
            "session_frac": min(max((hour - 12.0) / 8.0, 0.0), 1.0),
            # رژیم و بایاس
            "regime_up": int(rv == 1), "regime_range": int(rv == 0),
            "regime_down": int(rv == -1), "regime_chaos": int(rv == 2),
            "bias": int(bias[sig_i]),
        })

    ds = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)

    # فیچرهای توالی — فقط به گذشته نگاه می‌کنند (shift قبل از rolling)
    if len(ds):
        ds["hours_since_prev"] = (ds["entry_time"].diff()
                                  .dt.total_seconds() / 3600.0).fillna(1e3)
        prev_win = ds["win"].shift(1)
        prev_r = ds["r"].shift(1)
        ds["roll10_wr"] = prev_win.rolling(10, min_periods=3).mean() \
            .fillna(ds["win"].expanding().mean().shift(1)).fillna(0.35)
        ds["roll10_r"] = prev_r.rolling(10, min_periods=3).mean() \
            .fillna(ds["r"].expanding().mean().shift(1)).fillna(0.0)
    return ds.dropna(subset=[f for f in FEATURES if f in ds.columns] or ["win"]) \
        .reset_index(drop=True) if len(ds) else ds


# ------------------------------------------------------------------ #
# ۳) پنجره‌های ماهانه‌ی خالص‌شده (purged)
# ------------------------------------------------------------------ #
def purged_monthly_windows(ds: pd.DataFrame, is_months: int = 12,
                           oos_months: int = 1
                           ) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """مثل monthly_windows ولی روی معامله‌ها + purge:

    معامله‌های IS که خروج‌شان از شروع OOS عبور می‌کند از IS حذف
    می‌شوند (لیبل‌شان از قیمت‌های OOS ساخته شده).
    """
    keys = ds["entry_time"].dt.strftime("%Y-%m").to_numpy()
    out = []
    for is_mask, oos_mask in monthly_windows(keys, is_months, oos_months):
        is_idx = np.where(is_mask)[0]
        oos_idx = np.where(oos_mask)[0]
        if len(oos_idx) == 0:
            continue
        oos_start = ds["entry_time"].iloc[oos_idx].min()
        keep = np.array([ds["exit_time"].iloc[i] < oos_start
                         for i in is_idx])
        is_purged = is_idx[keep]
        out.append((is_purged, oos_idx,
                    ",".join(sorted(set(keys[oos_mask])))))
    return out


# ------------------------------------------------------------------ #
# ۴) walk-forward متا: انتخاب مدل/آستانه فقط با IS
# ------------------------------------------------------------------ #
@dataclass
class MetaWFOConfig:
    thresholds: tuple = (0.50, 0.55, 0.60)
    min_kept_is: int = 30          # کمتر از این = آن گزینه رد
    is_months: int = 12
    oos_months: int = 1
    seed: int = 42


@dataclass
class MetaWFOResult:
    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_take_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_meta: pd.DataFrame = field(default_factory=pd.DataFrame)


def _models(cfg: MetaWFOConfig) -> dict:
    return {
        "logit": Pipeline([("sc", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                                      random_state=cfg.seed))]),
        "gbm": GradientBoostingClassifier(max_depth=3, n_estimators=100,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=cfg.seed),
    }


def fit_probs(ds: pd.DataFrame, is_idx: np.ndarray,
              oos_idx: np.ndarray, cfg: MetaWFOConfig) -> dict:
    """فیت روی IS (خالص‌شده) → احتمال برد برای IS و OOS."""
    X = ds[FEATURES]
    y = ds["win"].to_numpy()
    probs = {}
    for name, mdl in _models(cfg).items():
        try:
            mdl.fit(X.iloc[is_idx], y[is_idx])
            probs[name] = (mdl.predict_proba(X.iloc[is_idx])[:, 1],
                           mdl.predict_proba(X.iloc[oos_idx])[:, 1])
        except ValueError:
            probs[name] = None
    return probs


def run_meta_wfo(ds: pd.DataFrame,
                 cfg: MetaWFOConfig | None = None) -> MetaWFOResult:
    """گرید = {take_all} ∪ {logit,gbm}×آستانه‌ها — انتخاب با IS-pnl.

    خروجی: معامله‌های OOS تحت (۱) همه‌ی کاندیداها (۲) فیلتر متا.
    """
    cfg = cfg or MetaWFOConfig()
    wins = purged_monthly_windows(ds, cfg.is_months, cfg.oos_months)
    rows, meta_frames, all_frames = [], [], []

    for is_idx, oos_idx, label in wins:
        if len(is_idx) < cfg.min_kept_is:
            continue
        is_df = ds.iloc[is_idx]
        oos_df = ds.iloc[oos_idx]
        is_pnl_all = float(is_df["pnl"].sum())
        best = ("take_all", None, is_pnl_all)

        probs = fit_probs(ds, is_idx, oos_idx, cfg)
        for name, pr in probs.items():
            if pr is None:
                continue
            p_is, _ = pr
            for th in cfg.thresholds:
                keep = p_is >= th
                n = int(keep.sum())
                if n < cfg.min_kept_is:
                    continue
                pnl = float(is_df["pnl"].to_numpy()[keep].sum())
                if pnl > best[2]:
                    best = ((f"{name}@{th:.2f}"), (name, th), pnl)

        # اعمال انتخاب منجمد روی OOS
        if best[1] is None:
            taken = oos_df
        else:
            name, th = best[1]
            p_oos = probs[name][1]
            taken = oos_df[p_oos >= th]

        meta_frames.append(taken)
        all_frames.append(oos_df)
        rows.append({
            "oos_month": label, "is_purged_n": len(is_idx),
            "chosen": best[0],
            "oos_taken": len(taken), "oos_total": len(oos_df),
            "oos_pnl": float(taken["pnl"].sum()),
            "oos_pnl_take_all": float(oos_df["pnl"].sum()),
        })

    per = pd.DataFrame(rows)
    return MetaWFOResult(
        per_window=per,
        oos_take_all=pd.concat(all_frames, ignore_index=True)
        if all_frames else pd.DataFrame(),
        oos_meta=pd.concat(meta_frames, ignore_index=True)
        if meta_frames else pd.DataFrame(),
    )


def agg(frame: pd.DataFrame, n_windows: int) -> dict:
    """جمع‌بندی یک مجموعه‌ی OOS: PF / WR / فرکانس / سود."""
    if not len(frame):
        return {"n": 0, "per_month": 0.0, "wr": float("nan"),
                "pf": float("nan"), "pnl": 0.0}
    gw = float(frame.loc[frame["pnl"] > 0, "pnl"].sum())
    gl = -float(frame.loc[frame["pnl"] <= 0, "pnl"].sum())
    return {"n": len(frame), "per_month": len(frame) / max(n_windows, 1),
            "wr": float((frame["pnl"] > 0).mean()),
            "pf": gw / gl if gl else float("inf"),
            "pnl": float(frame["pnl"].sum())}
