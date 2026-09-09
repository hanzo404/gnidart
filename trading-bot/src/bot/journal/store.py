"""ژورنال SQLite — حافظه بلندمدت ربات.

هر معامله با «فیچرهای لحظه ورود» ذخیره می‌شود؛ این دیتاست بعداً:
    1. گزارش‌های پست‌مورتم (طبقه‌بندی خطاها) — فاز ۴
    2. walk-forward ماهانه — فاز ۲
    3. meta-labeling (مدل فیلتر/حجم‌دهی سیگنال) — فاز ۶
را تغذیه می‌کند. بدون این دیتابیس، «خودتحلیلی» فقط شعار است.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at     TEXT NOT NULL,
    closed_at     TEXT,
    symbol        TEXT NOT NULL,
    direction     INTEGER NOT NULL,          -- +1 long / -1 short
    strategy      TEXT NOT NULL,
    regime        TEXT,                      -- trend | range | volatile | ...
    grade         TEXT NOT NULL,             -- A | B | C
    entry         REAL NOT NULL,
    stop          REAL NOT NULL,
    target        REAL,
    size_units    REAL NOT NULL,
    exit_price    REAL,
    r_multiple    REAL,                      -- نتیجه بر حسب R
    mfe_r         REAL,                      -- حداکثر سودِ دیده‌شده (در R)
    mae_r         REAL,                      -- حداکثر ضررِ دیده‌شده (در R)
    features_json TEXT,                      -- فیچرهای لحظه ورود (بدون نشتی آینده)
    reason        TEXT,
    status        TEXT NOT NULL DEFAULT 'open'  -- open | closed | cancelled
);
CREATE TABLE IF NOT EXISTS breaker_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    kind            TEXT NOT NULL,
    detail          TEXT,
    streak          INTEGER,
    size_multiplier REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts     TEXT NOT NULL,
    equity REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_breaker_ts ON breaker_events(ts);
"""


class Journal:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ----------------写入---------------- #
    def open_trade(
        self,
        opened_at: datetime,
        symbol: str,
        direction: int,
        strategy: str,
        grade: str,
        entry: float,
        stop: float,
        target: Optional[float],
        size_units: float,
        regime: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (opened_at, symbol, direction, strategy, regime, grade,
               entry, stop, target, size_units, features_json, reason, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open')""",
            (
                opened_at.isoformat(), symbol, direction, strategy, regime, grade,
                entry, stop, target, size_units,
                json.dumps(features or {}, ensure_ascii=False), reason,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def close_trade(
        self,
        trade_id: int,
        closed_at: datetime,
        exit_price: float,
        r_multiple: float,
        mfe_r: Optional[float] = None,
        mae_r: Optional[float] = None,
    ) -> None:
        self.conn.execute(
            """UPDATE trades SET closed_at=?, exit_price=?, r_multiple=?,
               mfe_r=?, mae_r=?, status='closed' WHERE id=?""",
            (closed_at.isoformat(), exit_price, r_multiple, mfe_r, mae_r, trade_id),
        )
        self.conn.commit()

    def record_breaker_event(self, ts: datetime, kind: str, detail: str,
                             streak: int, size_multiplier: float) -> None:
        self.conn.execute(
            "INSERT INTO breaker_events (ts, kind, detail, streak, size_multiplier) VALUES (?,?,?,?,?)",
            (ts.isoformat(), kind, detail, streak, size_multiplier),
        )
        self.conn.commit()

    def record_equity(self, ts: datetime, equity: float) -> None:
        self.conn.execute("INSERT INTO equity (ts, equity) VALUES (?,?)",
                          (ts.isoformat(), equity))
        self.conn.commit()

    # ----------------读取---------------- #
    def recent_trades(self, n: int = 50, status: str = "closed") -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE status=? ORDER BY closed_at DESC LIMIT ?",
            (status, n),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d.pop("features_json") or "{}")
            out.append(d)
        return out

    def stats(self) -> Dict[str, float]:
        """آمار سریع — پایه‌ی گزارش‌های خودتحلیلی."""
        row = self.conn.execute(
            """SELECT COUNT(*) n,
                      SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) wins,
                      AVG(r_multiple) avg_r,
                      SUM(r_multiple) total_r
               FROM trades WHERE status='closed'"""
        ).fetchone()
        n = row["n"] or 0
        return {
            "trades": float(n),
            "win_rate": (row["wins"] or 0) / n if n else 0.0,
            "avg_r": row["avg_r"] or 0.0,
            "total_r": row["total_r"] or 0.0,
        }

    def close(self) -> None:
        self.conn.close()
