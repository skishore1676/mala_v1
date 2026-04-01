"""
SQLite-backed results store for backtest artifacts.

Purpose:
- Keep CSV/JSON artifact files as immutable outputs.
- Also persist structured rows into a local DB for fast querying.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from src.config import DATA_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_non_none(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


class ResultsDB:
    """Persist strategy artifacts into a queryable SQLite database."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (DATA_DIR / "results" / "results.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    script TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    params_json TEXT
                );

                CREATE TABLE IF NOT EXISTS artifact_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    script TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ticker TEXT,
                    strategy TEXT,
                    direction TEXT,
                    decision TEXT,
                    cost_r REAL,
                    ratio REAL,
                    window_idx INTEGER,
                    signals INTEGER,
                    exp_r REAL,
                    confidence REAL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_artifact_rows_run_id
                    ON artifact_rows(run_id);
                CREATE INDEX IF NOT EXISTS idx_artifact_rows_artifact
                    ON artifact_rows(artifact_type);
                CREATE INDEX IF NOT EXISTS idx_artifact_rows_keydims
                    ON artifact_rows(ticker, strategy, direction);
                CREATE INDEX IF NOT EXISTS idx_artifact_rows_decision
                    ON artifact_rows(decision);

                CREATE TABLE IF NOT EXISTS research_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    ticker TEXT,
                    research_slice_id TEXT,
                    config_signature TEXT NOT NULL,
                    request_signature TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    total_signals INTEGER,
                    exp_r REAL,
                    confidence REAL,
                    avg_mfe_mae_ratio REAL,
                    effective_cost_r REAL,
                    runtime_seconds REAL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_research_evaluations_strategy
                    ON research_evaluations(strategy);
                CREATE INDEX IF NOT EXISTS idx_research_evaluations_signature
                    ON research_evaluations(config_signature);
                CREATE INDEX IF NOT EXISTS idx_research_evaluations_ticker
                    ON research_evaluations(ticker);
                """
            )
        self._ensure_research_evaluations_migrations()

    def _ensure_research_evaluations_migrations(self) -> None:
        with self._connect() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(research_evaluations)").fetchall()
            }
            if "research_slice_id" not in columns:
                conn.execute(
                    "ALTER TABLE research_evaluations ADD COLUMN research_slice_id TEXT"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_research_evaluations_slice
                    ON research_evaluations(research_slice_id)
                """
            )

    def start_run(self, script: str, params: dict[str, Any] | None = None) -> str:
        run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, script, started_at, params_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    script,
                    _utc_now(),
                    json.dumps(params or {}, default=str),
                ),
            )
        return run_id

    def finish_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?",
                (_utc_now(), run_id),
            )

    def ingest_dataframe(
        self,
        *,
        run_id: str,
        script: str,
        artifact_type: str,
        source_path: str,
        df: pl.DataFrame,
    ) -> None:
        if df.is_empty():
            return

        now = _utc_now()
        rows_to_insert: list[tuple[Any, ...]] = []
        for row in df.iter_rows(named=True):
            signals = _to_int(_first_non_none(row, [
                "signals",
                "oos_signals",
                "min_oos_signals",
                "total_signals",
                "holdout_trades",
                "test_signals",
                "holdout_signals",
                "min_holdout_signals",
            ]))
            exp_r = _to_float(_first_non_none(row, [
                "exp_r",
                "avg_test_exp_r",
                "test_exp_r",
                "holdout_exp_r",
                "min_holdout_exp_r",
                "min_avg_test_exp_r",
                "base_exp_r",
                "mc_exp_r_p50",
            ]))
            confidence = _to_float(_first_non_none(row, [
                "confidence",
                "confidence_2to1",
                "avg_test_confidence",
                "test_confidence",
                "holdout_confidence",
                "mean_test_confidence",
                "holdout_win_rate",
            ]))
            ratio = _to_float(_first_non_none(row, ["ratio", "selected_ratio"]))

            rows_to_insert.append(
                (
                    run_id,
                    script,
                    artifact_type,
                    source_path,
                    now,
                    row.get("ticker"),
                    row.get("strategy"),
                    row.get("direction"),
                    row.get("decision"),
                    _to_float(row.get("cost_r")),
                    ratio,
                    _to_int(row.get("window_idx")),
                    signals,
                    exp_r,
                    confidence,
                    json.dumps(row, default=str),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO artifact_rows(
                    run_id, script, artifact_type, source_path, created_at,
                    ticker, strategy, direction, decision,
                    cost_r, ratio, window_idx, signals, exp_r, confidence, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

    def fetch_research_evaluation(self, request_signature: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM research_evaluations
                WHERE request_signature = ?
                """,
                (request_signature,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def store_research_evaluation(self, payload: dict[str, Any]) -> None:
        constraints = payload.get("constraints", {})
        objective = payload.get("objective", {})
        tickers = payload.get("slice", {}).get("tickers") or []
        ticker = tickers[0] if len(tickers) == 1 else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_evaluations(
                    created_at,
                    strategy,
                    ticker,
                    research_slice_id,
                    config_signature,
                    request_signature,
                    status,
                    total_signals,
                    exp_r,
                    confidence,
                    avg_mfe_mae_ratio,
                    effective_cost_r,
                    runtime_seconds,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    payload.get("strategy"),
                    ticker,
                    payload.get("research_slice_id"),
                    payload.get("config_signature"),
                    payload.get("request_signature"),
                    payload.get("status"),
                    _to_int(constraints.get("total_signals")),
                    _to_float(objective.get("value")),
                    _to_float(objective.get("confidence")),
                    _to_float(objective.get("avg_mfe_mae_ratio")),
                    _to_float(constraints.get("effective_cost_r")),
                    _to_float(payload.get("runtime_seconds")),
                    json.dumps(payload, default=str),
                ),
            )

    def list_research_evaluations(
        self,
        *,
        strategy: str,
        ticker: str | None = None,
        research_slice_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT payload_json
            FROM research_evaluations
            WHERE strategy = ?
        """
        params: list[Any] = [strategy]
        if research_slice_id is not None:
            query += " AND research_slice_id = ?"
            params.append(research_slice_id)
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        payloads = [json.loads(row[0]) for row in rows]
        if ticker is None:
            return payloads
        filtered: list[dict[str, Any]] = []
        for payload in payloads:
            payload_tickers = payload.get("slice", {}).get("tickers") or []
            if ticker in payload_tickers:
                filtered.append(payload)
        return filtered
