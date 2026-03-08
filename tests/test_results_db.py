"""Tests for SQLite results persistence."""

from pathlib import Path
import sqlite3

import polars as pl

from src.oracle.results_db import ResultsDB


def test_results_db_ingests_dataframe(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path=db_path)
    run_id = db.start_run("test_script.py", params={"a": 1})

    df = pl.DataFrame([
        {
            "ticker": "SPY",
            "strategy": "Elastic Band Reversion",
            "direction": "short",
            "oos_signals": 500,
            "avg_test_exp_r": 0.12,
            "avg_test_confidence": 0.41,
            "decision": "candidate_needs_more_stability",
        }
    ])

    db.ingest_dataframe(
        run_id=run_id,
        script="test_script.py",
        artifact_type="walk_forward_novel_summary",
        source_path="data/results/fake.csv",
        df=df,
    )
    db.finish_run(run_id)

    with sqlite3.connect(db_path) as conn:
        run_rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        art_rows = conn.execute("SELECT COUNT(*) FROM artifact_rows").fetchone()[0]
        exp_r = conn.execute("SELECT exp_r FROM artifact_rows").fetchone()[0]

    assert run_rows == 1
    assert art_rows == 1
    assert abs(float(exp_r) - 0.12) < 1e-9


def test_results_db_keeps_zero_values(tmp_path: Path) -> None:
    db_path = tmp_path / "results_zero.db"
    db = ResultsDB(db_path=db_path)
    run_id = db.start_run("test_zero.py")

    df = pl.DataFrame([
        {"ticker": "SPY", "strategy": "X", "direction": "long", "confidence": 0.0, "exp_r": 0.0, "signals": 0}
    ])
    db.ingest_dataframe(
        run_id=run_id,
        script="test_zero.py",
        artifact_type="zero_case",
        source_path="zero.csv",
        df=df,
    )
    db.finish_run(run_id)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT confidence, exp_r, signals FROM artifact_rows").fetchone()

    assert row is not None
    assert float(row[0]) == 0.0
    assert float(row[1]) == 0.0
    assert int(row[2]) == 0
