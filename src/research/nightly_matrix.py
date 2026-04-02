"""Nightly regime-matrix orchestration for Mala."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from src.config import PROJECT_ROOT
from src.research.loop_contracts import (
    DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
    PLAYBOOK_CATALOG_CONTRACT_NAME,
    LOOP_ARTIFACT_SCHEMA_VERSION,
)
from src.research.loop_export import LoopArtifactExporter
from src.research.playbooks import augment_playbook_catalog_from_queue
from src.research.review_queue import HumanReviewQueueManager
from src.research.run_storage import create_run_dir


NIGHTLY_REGIME_MATRIX_CONFIG_VERSION = 1
_OUT_DIR_RE = re.compile(r"^OUT_DIR=(?P<path>.+)$", re.MULTILINE)
_FAMILY_TO_SCRIPT = {
    "market_impulse": "scripts/run_agentic_market_impulse_pass.py",
    "jerk_pivot_momentum": "scripts/run_agentic_jerk_pivot_pass.py",
    "elastic_band_reversion": "scripts/run_agentic_first_pass.py",
}


class NightlyResearchDefaults(BaseModel):
    start: date = date(2024, 1, 2)
    end: date = date(2026, 2, 28)
    calibration_end: date = date(2025, 11, 30)
    holdout_start: date = date(2025, 12, 1)
    holdout_end: date = date(2026, 2, 28)
    train_months: int = 6
    test_months: int = 3
    ratios: str = "1.0,1.25,1.5,2.0"
    m1_cost_bps: float = 8.0
    cost_grid_bps: str = "5,8,12"
    min_signals: int = 15
    gate_min_oos_windows: int = 6
    gate_min_oos_signals: int = 120
    gate_min_pct_positive: float = 0.55
    gate_min_exp_r: float = 0.0
    min_calibration_signals: int = 40
    min_holdout_signals: int = 20
    base_cost_r: float = 0.08
    bootstrap_iters: int = 4000
    top_per_ticker: int = 1
    broad_scout_max_stage: str = "M2"

    @model_validator(mode="after")
    def validate_defaults(self) -> "NightlyResearchDefaults":
        if self.broad_scout_max_stage not in {"M2", "M5"}:
            raise ValueError(
                "broad_scout_max_stage must be one of: M2, M5"
            )
        return self

    def cli_args(self) -> list[str]:
        return [
            "--start",
            self.start.isoformat(),
            "--end",
            self.end.isoformat(),
            "--calibration-end",
            self.calibration_end.isoformat(),
            "--holdout-start",
            self.holdout_start.isoformat(),
            "--holdout-end",
            self.holdout_end.isoformat(),
            "--train-months",
            str(self.train_months),
            "--test-months",
            str(self.test_months),
            "--ratios",
            self.ratios,
            "--m1-cost-bps",
            str(self.m1_cost_bps),
            "--cost-grid-bps",
            self.cost_grid_bps,
            "--min-signals",
            str(self.min_signals),
            "--gate-min-oos-windows",
            str(self.gate_min_oos_windows),
            "--gate-min-oos-signals",
            str(self.gate_min_oos_signals),
            "--gate-min-pct-positive",
            str(self.gate_min_pct_positive),
            "--gate-min-exp-r",
            str(self.gate_min_exp_r),
            "--min-calibration-signals",
            str(self.min_calibration_signals),
            "--min-holdout-signals",
            str(self.min_holdout_signals),
            "--base-cost-r",
            str(self.base_cost_r),
            "--bootstrap-iters",
            str(self.bootstrap_iters),
            "--top-per-ticker",
            str(self.top_per_ticker),
            "--max-stage",
            self.broad_scout_max_stage,
        ]


class NightlyFollowupBudgets(BaseModel):
    max_new_m2_rows_per_night: int = 10
    max_retune_tasks_per_night: int = 4
    max_symbol_expansion_tasks_per_night: int = 3
    max_m3_promotions_per_night: int = 3
    max_total_followup_tasks_per_night: int = 8


class NightlyRegimeMatrixConfig(BaseModel):
    schema_version: int = NIGHTLY_REGIME_MATRIX_CONFIG_VERSION
    output_root: str = "data/results/nightly_regime_matrix"
    research_control_root: str | None = None
    watchlist: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL"])
    tier2_watchlist: list[str] = Field(
        default_factory=lambda: ["META", "MSFT", "AMZN", "AMD", "SMH", "XLF"]
    )
    enabled_strategy_families: list[str] = Field(
        default_factory=lambda: [
            "market_impulse",
            "jerk_pivot_momentum",
            "elastic_band_reversion",
        ]
    )
    defaults: NightlyResearchDefaults = Field(default_factory=NightlyResearchDefaults)
    followup_budgets: NightlyFollowupBudgets = Field(default_factory=NightlyFollowupBudgets)

    @model_validator(mode="after")
    def validate_config(self) -> "NightlyRegimeMatrixConfig":
        self.watchlist = [symbol.upper() for symbol in self.watchlist]
        self.tier2_watchlist = [symbol.upper() for symbol in self.tier2_watchlist]
        if self.research_control_root is None:
            self.research_control_root = f"{self.output_root}/research_control"
        if self.schema_version != NIGHTLY_REGIME_MATRIX_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported nightly config schema_version: {self.schema_version}"
            )
        invalid = [
            family
            for family in self.enabled_strategy_families
            if family not in _FAMILY_TO_SCRIPT
        ]
        if invalid:
            raise ValueError(f"Unsupported strategy families: {invalid}")
        if not self.watchlist:
            raise ValueError("watchlist must not be empty")
        return self


@dataclass(slots=True, frozen=True)
class NightlyRegimeMatrixResult:
    bundle_dir: Path
    run_dirs: dict[str, Path]
    family_log_paths: dict[str, Path]
    deployment_candidates_path: Path
    playbook_catalog_path: Path
    playbook_projection_path: Path | None
    manifest_path: Path
    review_queue_path: Path
    review_history_path: Path
    review_workbook_path: Path
    review_bundle_dir: Path
    charts_dir: Path
    scout_only_run: bool
    deployment_candidates_generated: int
    followup_actions_run_count: int
    summary_reason: str


FamilyRunner = Callable[[str, NightlyRegimeMatrixConfig, Path], tuple[Path, Path]]


def load_nightly_regime_matrix_config(
    path: str | Path = PROJECT_ROOT / "config" / "nightly_regime_matrix.yaml",
) -> NightlyRegimeMatrixConfig:
    return NightlyRegimeMatrixConfig.model_validate(_read_yaml(Path(path)))


def run_nightly_regime_matrix(
    config: NightlyRegimeMatrixConfig,
    *,
    bundle_dir: str | Path | None = None,
    family_runner: FamilyRunner | None = None,
    exporter: LoopArtifactExporter | None = None,
    run_date: date | None = None,
    review_queue_manager: HumanReviewQueueManager | None = None,
) -> NightlyRegimeMatrixResult:
    resolved_bundle_dir = (
        Path(bundle_dir).resolve()
        if bundle_dir is not None
        else create_run_dir(PROJECT_ROOT / config.output_root, "nightly_regime_matrix").resolve()
    )
    resolved_bundle_dir.mkdir(parents=True, exist_ok=True)
    runner = family_runner or _run_family_research
    loop_exporter = exporter or LoopArtifactExporter()
    resolved_run_date = run_date or datetime.now().date()
    queue_manager = review_queue_manager or HumanReviewQueueManager(config.research_control_root)

    run_dirs: dict[str, Path] = {}
    family_log_paths: dict[str, Path] = {}
    for family in config.enabled_strategy_families:
        run_dir, log_path = runner(family, config, resolved_bundle_dir)
        run_dirs[family] = run_dir
        family_log_paths[family] = log_path

    candidates_path, playbook_path = loop_exporter.export_runs(
        list(run_dirs.values()),
        out_dir=resolved_bundle_dir,
        watchlist=config.watchlist,
        enabled_strategy_families=config.enabled_strategy_families,
    )
    review_artifacts = queue_manager.refresh_queue(
        run_dirs=run_dirs,
        config=config,
        run_date=resolved_run_date,
    )
    playbook_path, playbook_projection_path = augment_playbook_catalog_from_queue(
        playbook_catalog_path=playbook_path,
        queue_path=review_artifacts.queue_path,
        playbook_projection_path=resolved_bundle_dir / "playbook_catalog.csv",
    )
    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    scout_only_run = config.defaults.broad_scout_max_stage == "M2"
    deployment_candidates_generated = len(candidates_payload.get("candidates", []))
    followup_actions_run_count = review_artifacts.followup_actions_run_count
    summary_reason = _nightly_summary_reason(
        scout_only_run=scout_only_run,
        deployment_candidates_generated=deployment_candidates_generated,
        followup_actions_run_count=followup_actions_run_count,
    )

    manifest_path = resolved_bundle_dir / "nightly_matrix_manifest.json"
    manifest_payload = {
        "config_schema_version": NIGHTLY_REGIME_MATRIX_CONFIG_VERSION,
        "config_watchlist": config.watchlist,
        "config_tier2_watchlist": config.tier2_watchlist,
        "enabled_strategy_families": config.enabled_strategy_families,
        "bundle_dir": str(resolved_bundle_dir),
        "run_dirs": {family: str(path) for family, path in run_dirs.items()},
        "family_logs": {family: str(path) for family, path in family_log_paths.items()},
        "review_control": {
            "queue_path": str(review_artifacts.queue_path),
            "history_path": str(review_artifacts.history_path),
            "workbook_path": str(review_artifacts.workbook_path),
            "review_bundle_dir": str(review_artifacts.review_bundle_dir),
            "charts_dir": str(review_artifacts.charts_dir),
        },
        "nightly_summary": {
            "scout_only_run": scout_only_run,
            "deployment_candidates_generated": deployment_candidates_generated,
            "followup_actions_run_count": followup_actions_run_count,
            "reason": summary_reason,
        },
        "contracts": {
            DEPLOYMENT_CANDIDATES_CONTRACT_NAME: {
                "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
                "path": str(candidates_path),
            },
            PLAYBOOK_CATALOG_CONTRACT_NAME: {
                "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
                "path": str(playbook_path),
                "projection_csv_path": str(playbook_projection_path) if playbook_projection_path is not None else None,
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NightlyRegimeMatrixResult(
        bundle_dir=resolved_bundle_dir,
        run_dirs=run_dirs,
        family_log_paths=family_log_paths,
        deployment_candidates_path=candidates_path,
        playbook_catalog_path=playbook_path,
        playbook_projection_path=playbook_projection_path,
        manifest_path=manifest_path,
        review_queue_path=review_artifacts.queue_path,
        review_history_path=review_artifacts.history_path,
        review_workbook_path=review_artifacts.workbook_path,
        review_bundle_dir=review_artifacts.review_bundle_dir,
        charts_dir=review_artifacts.charts_dir,
        scout_only_run=scout_only_run,
        deployment_candidates_generated=deployment_candidates_generated,
        followup_actions_run_count=followup_actions_run_count,
        summary_reason=summary_reason,
    )


def _run_family_research(
    family: str,
    config: NightlyRegimeMatrixConfig,
    bundle_dir: Path,
) -> tuple[Path, Path]:
    script_relative = _FAMILY_TO_SCRIPT[family]
    script_path = (PROJECT_ROOT / script_relative).resolve()
    family_root = bundle_dir / "family_runs"
    family_root.mkdir(parents=True, exist_ok=True)
    logs_root = bundle_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{family}.log"
    command = [
        sys.executable,
        str(script_path),
        "--tickers",
        *config.watchlist,
        *config.defaults.cli_args(),
        "--out-dir",
        str(family_root),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    output_text = log_path.read_text(encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nightly family runner failed for {family}\n"
            f"See log: {log_path}\n"
            f"Log tail:\n{_tail_text(output_text)}"
        )
    match = _OUT_DIR_RE.search(output_text)
    if match is None:
        raise RuntimeError(
            f"Could not locate OUT_DIR in {family} runner output.\n"
            f"See log: {log_path}\n"
            f"Log tail:\n{_tail_text(output_text)}"
        )
    return Path(match.group("path")).resolve(), log_path.resolve()


def _tail_text(text: str, *, line_count: int = 40) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "<empty log>"
    return "\n".join(lines[-line_count:])


def _nightly_summary_reason(
    *,
    scout_only_run: bool,
    deployment_candidates_generated: int,
    followup_actions_run_count: int,
) -> str:
    if scout_only_run and deployment_candidates_generated == 0 and followup_actions_run_count == 0:
        return "no M3-M5 follow-up executed"
    if scout_only_run and deployment_candidates_generated == 0:
        return "broad scout stopped at M2; validated follow-up exports were not generated into the top-level deployment bundle"
    if deployment_candidates_generated > 0:
        return "validated follow-up artifacts produced deployable playbooks"
    return "no deployable candidates were exported"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        ruby_cmd = [
            "ruby",
            "-e",
            "require 'yaml'; require 'json'; puts JSON.generate(YAML.load_file(ARGV[0]))",
            str(path),
        ]
        completed = subprocess.run(
            ruby_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)  # type: ignore[attr-defined]
    return data or {}


__all__ = [
    "NIGHTLY_REGIME_MATRIX_CONFIG_VERSION",
    "NightlyRegimeMatrixConfig",
    "NightlyRegimeMatrixResult",
    "NightlyResearchDefaults",
    "load_nightly_regime_matrix_config",
    "run_nightly_regime_matrix",
]
