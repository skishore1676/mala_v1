"""Tests for stage-by-stage research reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.research.models import ResearchStage
from src.research.reporting import ResearchJournal
from src.research.tools import ResearchToolResult


def test_research_journal_records_stage_manifest_and_markdown(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    journal = ResearchJournal(out_dir)
    result = ResearchToolResult(
        tool_name="parameter_sweep",
        summary={"top_candidates": 2},
        artifacts={
            "aggregate": pl.DataFrame(
                [{"ticker": "SPY", "strategy": "Stub", "direction": "short", "avg_test_exp_r": 0.1}]
            )
        },
    )

    paths = journal.record_stage(
        stage=ResearchStage.M1_DISCOVERY,
        result=result,
        decision="promote",
        rationale="A bounded sweep found enough signal to justify convergence.",
        next_action="Run convergence grid.",
        context={"strategy_family": "Stub"},
    )

    assert (out_dir / "M1_parameter_sweep.md").exists()
    assert (out_dir / "research_manifest.json").exists()
    assert "aggregate" in paths
    assert paths["aggregate"].exists()

    manifest = json.loads((out_dir / "research_manifest.json").read_text(encoding="utf-8"))
    stage_entry = manifest["stages"][0]
    assert stage_entry["stage"] == "M1"
    assert stage_entry["decision"] == "promote"
    assert stage_entry["next_action"] == "Run convergence grid."
