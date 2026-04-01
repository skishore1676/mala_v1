"""Nightly regime-matrix orchestration for Mala."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
        ]


class NightlyRegimeMatrixConfig(BaseModel):
    schema_version: int = NIGHTLY_REGIME_MATRIX_CONFIG_VERSION
    output_root: str = "data/results/nightly_regime_matrix"
    watchlist: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "TSLA", "PLTR"])
    enabled_strategy_families: list[str] = Field(
        default_factory=lambda: [
            "market_impulse",
            "jerk_pivot_momentum",
            "elastic_band_reversion",
        ]
    )
    defaults: NightlyResearchDefaults = Field(default_factory=NightlyResearchDefaults)

    @model_validator(mode="after")
    def validate_config(self) -> "NightlyRegimeMatrixConfig":
        self.watchlist = [symbol.upper() for symbol in self.watchlist]
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
    deployment_candidates_path: Path
    playbook_catalog_path: Path
    manifest_path: Path


FamilyRunner = Callable[[str, NightlyRegimeMatrixConfig, Path], Path]


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
) -> NightlyRegimeMatrixResult:
    resolved_bundle_dir = (
        Path(bundle_dir).resolve()
        if bundle_dir is not None
        else create_run_dir(PROJECT_ROOT / config.output_root, "nightly_regime_matrix").resolve()
    )
    resolved_bundle_dir.mkdir(parents=True, exist_ok=True)
    runner = family_runner or _run_family_research
    loop_exporter = exporter or LoopArtifactExporter()

    run_dirs: dict[str, Path] = {}
    for family in config.enabled_strategy_families:
        run_dirs[family] = runner(family, config, resolved_bundle_dir)

    candidates_path, playbook_path = loop_exporter.export_runs(
        list(run_dirs.values()),
        out_dir=resolved_bundle_dir,
        watchlist=config.watchlist,
        enabled_strategy_families=config.enabled_strategy_families,
    )

    manifest_path = resolved_bundle_dir / "nightly_matrix_manifest.json"
    manifest_payload = {
        "config_schema_version": NIGHTLY_REGIME_MATRIX_CONFIG_VERSION,
        "config_watchlist": config.watchlist,
        "enabled_strategy_families": config.enabled_strategy_families,
        "bundle_dir": str(resolved_bundle_dir),
        "run_dirs": {family: str(path) for family, path in run_dirs.items()},
        "contracts": {
            DEPLOYMENT_CANDIDATES_CONTRACT_NAME: {
                "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
                "path": str(candidates_path),
            },
            PLAYBOOK_CATALOG_CONTRACT_NAME: {
                "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
                "path": str(playbook_path),
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
        deployment_candidates_path=candidates_path,
        playbook_catalog_path=playbook_path,
        manifest_path=manifest_path,
    )


def _run_family_research(
    family: str,
    config: NightlyRegimeMatrixConfig,
    bundle_dir: Path,
) -> Path:
    script_relative = _FAMILY_TO_SCRIPT[family]
    script_path = (PROJECT_ROOT / script_relative).resolve()
    family_root = bundle_dir / "family_runs"
    family_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--tickers",
        *config.watchlist,
        *config.defaults.cli_args(),
        "--out-dir",
        str(family_root),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nightly family runner failed for {family}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    match = _OUT_DIR_RE.search(completed.stdout)
    if match is None:
        raise RuntimeError(
            f"Could not locate OUT_DIR in {family} runner output.\n"
            f"STDOUT:\n{completed.stdout}"
        )
    return Path(match.group("path")).resolve()


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
