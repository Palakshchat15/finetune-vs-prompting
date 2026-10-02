"""SQLite request tracing (data/traces.db). One row per classification request."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,              -- app | eval
    system TEXT,              -- tfidf_logreg | zero_shot_7b | few_shot_7b | base_1p5b | finetuned_1p5b
    model TEXT,
    input TEXT,
    prediction TEXT,
    valid INTEGER,            -- 1 if the prediction is one of the 77 labels
    raw_output TEXT,
    t_retrieve REAL,          -- kNN example retrieval (few-shot only)
    t_model REAL,             -- model call wall time
    t_total REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    error TEXT
);
"""


class Tracer:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def log(self, **kw) -> int:
        row = {"ts": time.time(), **kw}
        cols = ", ".join(row)
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO traces ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
            return int(cur.lastrowid)

    def read(self, limit: int | None = None):
        import pandas as pd

        q = "SELECT * FROM traces ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            return pd.read_sql_query(q, c)
