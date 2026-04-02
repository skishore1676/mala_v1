"""Export Mala research runs into Bhiksha loop artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import polars as pl
from polars.exceptions import NoDataError

from src.config import PROJECT_ROOT
from src.research.loop_contracts import (
    DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
    PLAYBOOK_CATALOG_CONTRACT_NAME,
    build_contract_metadata,
)
from src.research.stages.candidates import candidate_params
from src.strategy.factory import build_strategy, build_strategy_by_name


_SUPPORTED_EXECUTION_PRESETS: dict[str, dict[str, Any]] = {
    "market_impulse": {
        "profile": "single_leg_long_premium_v1",
        "shadow_only": True,
        "option_mapping": {
            "long_signal": "CALL",
            "short_signal": "PUT",
        },
        "dte_min": 0,
        "dte_max": 7,
        "target_abs_delta_min": 0.20,
        "target_abs_delta_max": 0.40,
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.20,
    },
    "jerk_pivot_momentum": {
        "profile": "single_leg_long_premium_v1",
        "shadow_only": True,
        "option_mapping": {
            "long_signal": "CALL",
            "short_signal": "PUT",
        },
        "dte_min": 7,
        "dte_max": 21,
        "target_abs_delta_min": 0.35,
        "target_abs_delta_max": 0.55,
        "entry_window_start_et": "09:45",
        "entry_window_end_et": "14:30",
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.20,
    },
    "elastic_band_reversion": {
        "profile": "single_leg_long_premium_v1",
        "shadow_only": True,
        "option_mapping": {
            "long_signal": "CALL",
            "short_signal": "PUT",
        },
        "dte_min": 7,
        "dte_max": 21,
        "target_abs_delta_min": 0.30,
        "target_abs_delta_max": 0.50,
        "entry_window_start_et": "09:45",
        "entry_window_end_et": "14:30",
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.20,
    },
    "opening_drive_classifier": {
        "profile": "single_leg_long_premium_v1",
        "shadow_only": True,
        "option_mapping": {
            "long_signal": "CALL",
            "short_signal": "PUT",
        },
        "dte_min": 0,
        "dte_max": 7,
        "target_abs_delta_min": 0.25,
        "target_abs_delta_max": 0.45,
        "entry_window_start_et": "09:55",
        "entry_window_end_et": "11:30",
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.20,
    },
}

_SUPPORTED_RISK_PRESETS: dict[str, dict[str, Any]] = {
    "market_impulse": {
        "profile": "conservative_day1",
        "max_trade_premium_usd": 300,
        "hard_flat_time_et": "15:55",
        "stop_loss_pct": 0.45,
    },
    "jerk_pivot_momentum": {
        "profile": "conservative_day1",
        "max_trade_premium_usd": 300,
        "hard_flat_time_et": "15:55",
        "stop_loss_pct": 0.35,
    },
    "elastic_band_reversion": {
        "profile": "conservative_day1",
        "max_trade_premium_usd": 300,
        "hard_flat_time_et": "15:55",
        "stop_loss_pct": 0.45,
    },
    "opening_drive_classifier": {
        "profile": "conservative_day1",
        "max_trade_premium_usd": 300,
        "hard_flat_time_et": "15:55",
        "stop_loss_pct": 0.40,
    },
}

_SUPPORTED_EXIT_PRESETS: dict[str, dict[str, Any]] = {
    "market_impulse": {
        "profile": "market_impulse_exit_v1",
        "use_algorithmic_exit": True,
        "use_profit_target": False,
        "profit_target_multiple": None,
        "stop_loss_pct": 0.45,
        "stop_to_breakeven_after_r_multiple": None,
        "hard_flat_time_et": "15:55",
    },
    "jerk_pivot_momentum": {
        "profile": "premium_target_v1",
        "use_algorithmic_exit": False,
        "use_profit_target": True,
        "profit_target_multiple": 2.0,
        "stop_loss_pct": 0.35,
        "stop_to_breakeven_after_r_multiple": None,
        "hard_flat_time_et": "15:55",
    },
    "elastic_band_reversion": {
        "profile": "elastic_band_exit_v1",
        "use_algorithmic_exit": False,
        "use_profit_target": False,
        "profit_target_multiple": None,
        "stop_loss_pct": 0.45,
        "stop_to_breakeven_after_r_multiple": None,
        "hard_flat_time_et": "15:55",
    },
    "opening_drive_classifier": {
        "profile": "opening_drive_exit_v1",
        "use_algorithmic_exit": False,
        "use_profit_target": False,
        "profit_target_multiple": None,
        "stop_loss_pct": 0.40,
        "stop_to_breakeven_after_r_multiple": None,
        "hard_flat_time_et": "15:55",
    },
}

_PROPOSED_EXECUTION_DEFAULTS: dict[str, Any] = {
    "profile": "manual_research_only",
    "shadow_only": True,
    "option_mapping": {
        "long_signal": "CALL",
        "short_signal": "PUT",
    },
    "min_open_interest": 0,
}

_PROPOSED_RISK_DEFAULTS: dict[str, Any] = {
    "profile": "manual_research_only",
    "max_trade_premium_usd": 300,
    "hard_flat_time_et": "15:55",
    "stop_loss_pct": 0.45,
}

_PROPOSED_EXIT_DEFAULTS: dict[str, Any] = {
    "profile": "manual_research_only",
    "use_algorithmic_exit": False,
    "use_profit_target": False,
    "profit_target_multiple": None,
    "stop_loss_pct": 0.45,
    "stop_to_breakeven_after_r_multiple": None,
    "hard_flat_time_et": "15:55",
}

_M5_METRIC_FIELDS = {
    "selected_ratio",
    "holdout_trades",
    "holdout_win_rate",
    "base_exp_r",
    "trades",
    "mc_exp_r_mean",
    "mc_exp_r_p05",
    "mc_exp_r_p50",
    "mc_exp_r_p95",
    "mc_prob_positive_exp",
    "mc_total_r_p05",
    "mc_total_r_p50",
    "mc_total_r_p95",
    "mc_max_dd_p50",
    "structure",
    "dte",
    "delta_plan",
    "entry_window_et",
    "profit_take",
    "risk_rule",
    "execution_profile",
    "stress_profile",
}

_M4_SUMMARY_METRIC_FIELDS = {
    "observed_cost_points",
    "min_holdout_signals",
    "min_holdout_exp_r",
    "mean_holdout_exp_r",
    "passes_all_cost_gates",
    "passes_holdout",
    "decision",
}

_BIAS_CONTEXTS = {
    ("market_impulse", "long"): "bullish_trend_intraday",
    ("market_impulse", "short"): "bearish_trend_intraday",
    ("jerk_pivot_momentum", "long"): "bullish_trend_intraday",
    ("jerk_pivot_momentum", "short"): "bearish_trend_intraday",
    ("elastic_band_reversion", "long"): "bullish_mean_reversion_intraday",
    ("elastic_band_reversion", "short"): "bearish_mean_reversion_intraday",
}

_ALL_BIAS_TEMPLATES = (
    "bullish_trend_intraday",
    "bullish_mean_reversion_intraday",
    "bearish_trend_intraday",
    "bearish_mean_reversion_intraday",
)

_CONTEXT_FAMILY_MAP = {
    "bullish_trend_intraday": ("market_impulse", "jerk_pivot_momentum", "opening_drive_classifier"),
    "bullish_mean_reversion_intraday": ("elastic_band_reversion", "opening_drive_classifier"),
    "bearish_trend_intraday": ("market_impulse", "jerk_pivot_momentum", "opening_drive_classifier"),
    "bearish_mean_reversion_intraday": ("elastic_band_reversion", "opening_drive_classifier"),
}


@dataclass(slots=True, frozen=True)
class ExportCandidate:
    candidate_id: str
    surface_class: str
    automation_status: str
    strategy_key: str
    symbol: str
    direction: str
    bias_template: str
    horizon: str
    ranking_score: float
    manifest: dict[str, Any]
    evidence: dict[str, Any]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "surface_class": self.surface_class,
            "automation_status": self.automation_status,
            "strategy_key": self.strategy_key,
            "symbol": self.symbol,
            "direction": self.direction,
            "bias_template": self.bias_template,
            "horizon": self.horizon,
            "ranking_score": self.ranking_score,
            "manifest": self.manifest,
            "evidence": self.evidence,
            "source": self.source,
        }


class LoopArtifactExporter:
    """Compile research runs into deployment and playbook artifacts."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self._supported_surface, self._proposed_surface = self._load_surface_docs()

    def export_runs(
        self,
        run_dirs: list[str | Path],
        *,
        out_dir: str | Path,
        watchlist: list[str] | None = None,
        enabled_strategy_families: list[str] | None = None,
    ) -> tuple[Path, Path]:
        resolved_run_dirs = [Path(run_dir).resolve() for run_dir in run_dirs]
        candidates: list[ExportCandidate] = []
        for run_dir in resolved_run_dirs:
            candidates.extend(self._collect_run_candidates(run_dir))
        deduped = self._dedupe_candidates(candidates)
        resolved_watchlist = self._resolve_watchlist(deduped, watchlist)
        resolved_families = self._resolve_enabled_strategy_families(deduped, enabled_strategy_families)
        generated_at = datetime.now(UTC).isoformat()
        candidates_payload = {
            **build_contract_metadata(DEPLOYMENT_CANDIDATES_CONTRACT_NAME),
            "generated_at": generated_at,
            "run_dirs": [self._relpath(path) for path in resolved_run_dirs],
            "watchlist": resolved_watchlist,
            "enabled_strategy_families": resolved_families,
            "candidates": [candidate.to_dict() for candidate in deduped],
        }
        playbook_payload = {
            **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
            "generated_at": generated_at,
            "run_dirs": [self._relpath(path) for path in resolved_run_dirs],
            "watchlist": resolved_watchlist,
            "enabled_strategy_families": resolved_families,
            "contexts": self._build_playbook_contexts(
                deduped,
                watchlist=resolved_watchlist,
                enabled_strategy_families=resolved_families,
            ),
        }

        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = target_dir / "deployment_candidates.json"
        playbook_path = target_dir / "playbook_catalog.json"
        candidates_path.write_text(json.dumps(candidates_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        playbook_path.write_text(json.dumps(playbook_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return candidates_path, playbook_path

    def _collect_run_candidates(self, run_dir: Path) -> list[ExportCandidate]:
        manifest_path = run_dir / "research_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing research manifest at {manifest_path}")
        research_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        m4_entry = self._stage_entry(research_manifest, "M4", required=False)
        m5_entry = self._stage_entry(research_manifest, "M5", required=False)
        if m4_entry is None or m5_entry is None:
            return []
        m4_summary = self._load_csv(run_dir, m4_entry, "summary")
        m4_detail = self._load_csv(run_dir, m4_entry, "detail")
        m5_detail = self._load_csv(run_dir, m5_entry, "detail")

        if m5_detail.is_empty():
            return []

        candidates: list[ExportCandidate] = []
        for row in m5_detail.iter_rows(named=True):
            strategy_name = str(row["strategy"])
            descriptor = _strategy_descriptor(strategy_name)
            surface_class = self._classify_surface(descriptor.strategy_key)
            strategy_params = self._strategy_params(strategy_name, row)
            bias_template = _bias_template(
                descriptor.strategy_key,
                str(row["direction"]),
                strategy_params=strategy_params,
            )
            manifest = self._build_manifest(
                descriptor=descriptor,
                symbol=str(row["ticker"]),
                direction=str(row["direction"]),
                strategy_params=strategy_params,
                surface_class=surface_class,
                execution_row=row,
            )
            signature_payload = {
                "surface_class": surface_class,
                "strategy_key": descriptor.strategy_key,
                "symbol": str(row["ticker"]).upper(),
                "direction": str(row["direction"]),
                "strategy": manifest["strategy"],
                "execution": manifest["execution"],
                "risk": manifest["risk"],
                "exit": manifest["exit"],
            }
            candidate_id = self._candidate_id(signature_payload)
            deployment_id = self._deployment_id(descriptor.strategy_key, str(row["ticker"]), str(row["direction"]), candidate_id)
            ranking_score = _ranking_score(row, m4_summary)
            automation_status = self._automation_status(surface_class=surface_class, m5_decision=str(m5_entry["decision"]))

            summary_row = self._matching_summary_row(m4_summary, row)
            detail_rows = self._matching_detail_rows(m4_detail, row)
            required_capabilities = (
                self._required_capabilities(descriptor.strategy_key)
                if surface_class == "proposed"
                else []
            )

            source = {
                "run_date": _stage_run_date(m5_entry),
                "research_manifest": self._relpath(manifest_path),
                "artifact": self._artifact_path(m5_entry, "detail"),
            }
            manifest["deployment_id"] = deployment_id
            manifest["enabled"] = True
            manifest["source"] = {
                "origin": "mala_loop_v1_1",
                "run_date": source["run_date"],
                "artifact": source["artifact"],
                "metadata": {
                    "candidate_id": candidate_id,
                    "surface_class": surface_class,
                    "automation_status": automation_status,
                    "bias_template": bias_template,
                    "horizon": "intraday",
                    "strategy_display_name": strategy_name,
                    "ranking_score": ranking_score,
                    "m5_structure": row.get("structure"),
                    "m5_delta_plan": row.get("delta_plan"),
                    "m5_profit_take": row.get("profit_take"),
                    "m5_risk_rule": row.get("risk_rule"),
                },
            }
            if required_capabilities:
                manifest["source"]["metadata"]["required_bhiksha_capabilities"] = required_capabilities
            evidence = {
                "strategy_display_name": strategy_name,
                "strategy_params": strategy_params,
                "holdout": {
                    "summary": summary_row,
                    "cost_points": detail_rows,
                },
                "execution": {
                    "selected_ratio": _maybe_float(row.get("selected_ratio")),
                    "base_exp_r": _maybe_float(row.get("base_exp_r")),
                    "structure": row.get("structure"),
                    "dte": row.get("dte"),
                    "delta_plan": row.get("delta_plan"),
                    "entry_window_et": row.get("entry_window_et"),
                    "profit_take": row.get("profit_take"),
                    "risk_rule": row.get("risk_rule"),
                },
                "monte_carlo": {
                    "mc_exp_r_mean": _maybe_float(row.get("mc_exp_r_mean")),
                    "mc_exp_r_p05": _maybe_float(row.get("mc_exp_r_p05")),
                    "mc_exp_r_p50": _maybe_float(row.get("mc_exp_r_p50")),
                    "mc_exp_r_p95": _maybe_float(row.get("mc_exp_r_p95")),
                    "mc_prob_positive_exp": _maybe_float(row.get("mc_prob_positive_exp")),
                    "mc_total_r_p05": _maybe_float(row.get("mc_total_r_p05")),
                    "mc_total_r_p50": _maybe_float(row.get("mc_total_r_p50")),
                    "mc_total_r_p95": _maybe_float(row.get("mc_total_r_p95")),
                    "mc_max_dd_p50": _maybe_float(row.get("mc_max_dd_p50")),
                },
                "sample_counts": {
                    "holdout_trades": _maybe_int(row.get("holdout_trades")),
                    "execution_trades": _maybe_int(row.get("trades")),
                },
                "artifact_paths": {
                    "research_manifest": self._relpath(manifest_path),
                    "m4_summary": self._artifact_path(m4_entry, "summary"),
                    "m4_detail": self._artifact_path(m4_entry, "detail"),
                    "m5_detail": self._artifact_path(m5_entry, "detail"),
                },
                "m5_decision": str(m5_entry["decision"]),
            }
            candidates.append(
                ExportCandidate(
                    candidate_id=candidate_id,
                    surface_class=surface_class,
                    automation_status=automation_status,
                    strategy_key=descriptor.strategy_key,
                    symbol=str(row["ticker"]).upper(),
                    direction=str(row["direction"]),
                    bias_template=bias_template,
                    horizon="intraday",
                    ranking_score=ranking_score,
                    manifest=manifest,
                    evidence=evidence,
                    source=source,
                )
            )
        return candidates

    def _dedupe_candidates(self, candidates: list[ExportCandidate]) -> list[ExportCandidate]:
        deduped: dict[str, ExportCandidate] = {}
        for candidate in candidates:
            existing = deduped.get(candidate.candidate_id)
            if existing is None:
                deduped[candidate.candidate_id] = candidate
                continue
            if self._candidate_sort_key(candidate) > self._candidate_sort_key(existing):
                deduped[candidate.candidate_id] = candidate
        return sorted(
            deduped.values(),
            key=lambda candidate: (
                candidate.symbol,
                candidate.bias_template,
                candidate.surface_class,
                -candidate.ranking_score,
                candidate.candidate_id,
            ),
        )

    def _candidate_sort_key(self, candidate: ExportCandidate) -> tuple[str, float]:
        return (candidate.source["run_date"], candidate.ranking_score)

    def _build_playbook_contexts(
        self,
        candidates: list[ExportCandidate],
        *,
        watchlist: list[str],
        enabled_strategy_families: list[str],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[ExportCandidate]] = {}
        for candidate in candidates:
            key = self._context_key(candidate.symbol, candidate.bias_template, candidate.horizon)
            grouped.setdefault(key, []).append(candidate)

        contexts: dict[str, dict[str, Any]] = {}
        for symbol in sorted(item.upper() for item in watchlist):
            for bias_template in _ALL_BIAS_TEMPLATES:
                key = self._context_key(symbol, bias_template, "intraday")
                grouped_candidates = grouped.get(key, [])
                supported = sorted(
                    (candidate for candidate in grouped_candidates if candidate.surface_class == "supported"),
                    key=lambda candidate: (-candidate.ranking_score, candidate.candidate_id),
                )
                proposed = sorted(
                    (candidate for candidate in grouped_candidates if candidate.surface_class == "proposed"),
                    key=lambda candidate: (-candidate.ranking_score, candidate.candidate_id),
                )
                covered_by_families = [
                    family
                    for family in _CONTEXT_FAMILY_MAP[bias_template]
                    if family in enabled_strategy_families
                ]
                coverage_status = _coverage_status(
                    supported=supported,
                    proposed=proposed,
                    covered_by_families=covered_by_families,
                )
                contexts[key] = {
                    "symbol": symbol,
                    "bias_template": bias_template,
                    "horizon": "intraday",
                    "coverage_status": coverage_status,
                    "covered_by_strategy_families": covered_by_families,
                    "supported_candidates": [self._playbook_entry(candidate, rank + 1) for rank, candidate in enumerate(supported)],
                    "proposed_candidates": [self._playbook_entry(candidate, rank + 1) for rank, candidate in enumerate(proposed)],
                }
        return contexts

    def _playbook_entry(self, candidate: ExportCandidate, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "candidate_id": candidate.candidate_id,
            "deployment_id": candidate.manifest["deployment_id"],
            "strategy_key": candidate.strategy_key,
            "symbol": candidate.symbol,
            "direction": candidate.direction,
            "surface_class": candidate.surface_class,
            "automation_status": candidate.automation_status,
            "ranking_score": candidate.ranking_score,
            "run_date": candidate.source["run_date"],
            "m5_decision": candidate.evidence["m5_decision"],
            "mc_prob_positive_exp": candidate.evidence["monte_carlo"]["mc_prob_positive_exp"],
            "mc_exp_r_p50": candidate.evidence["monte_carlo"]["mc_exp_r_p50"],
            "mean_holdout_exp_r": candidate.evidence["holdout"]["summary"].get("mean_holdout_exp_r"),
        }

    def _load_surface_docs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            _read_yaml(self.project_root / "docs" / "strategy_surface.yaml"),
            _read_yaml(self.project_root / "docs" / "strategy_surface_proposed.yaml"),
        )

    def _classify_surface(self, strategy_key: str) -> str:
        if strategy_key in set((self._supported_surface.get("strategy_families") or {}).keys()):
            return "supported"
        if strategy_key in set((self._proposed_surface.get("proposed_strategy_families") or {}).keys()):
            return "proposed"
        return "blocked"

    def _required_capabilities(self, strategy_key: str) -> list[str]:
        family = (self._proposed_surface.get("proposed_strategy_families") or {}).get(strategy_key, {})
        capabilities = family.get("required_bhiksha_capabilities") or []
        return [str(item) for item in capabilities]

    def _automation_status(self, *, surface_class: str, m5_decision: str) -> str:
        if surface_class == "supported" and m5_decision == "promote":
            return "shadow_ready"
        if surface_class == "proposed":
            return "manual_research_only"
        return "blocked"

    def _strategy_params(self, strategy_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if strategy_name.startswith("Elastic Band "):
            strategy = build_strategy_by_name(strategy_name)
            return strategy.strategy_config()
        params = candidate_params(row)
        strategy = build_strategy(strategy_name, params)
        config = strategy.strategy_config()
        config.pop("strategy_label", None)
        return config

    def _build_manifest(
        self,
        *,
        descriptor: "_StrategyDescriptor",
        symbol: str,
        direction: str,
        strategy_params: dict[str, Any],
        surface_class: str,
        execution_row: dict[str, Any],
    ) -> dict[str, Any]:
        if surface_class == "supported":
            return {
                "symbol": symbol.upper(),
                "strategy": {
                    "key": descriptor.strategy_key,
                    "version": 1,
                    "params": {
                        **strategy_params,
                        "direction": direction,
                    },
                },
                "execution": dict(_SUPPORTED_EXECUTION_PRESETS[descriptor.strategy_key]),
                "risk": dict(_SUPPORTED_RISK_PRESETS[descriptor.strategy_key]),
                "exit": dict(_SUPPORTED_EXIT_PRESETS[descriptor.strategy_key]),
            }

        execution = dict(_PROPOSED_EXECUTION_DEFAULTS)
        dte_min, dte_max = _parse_range(execution_row.get("dte"))
        if dte_min is not None:
            execution["dte_min"] = dte_min
            execution["dte_max"] = dte_max
        window_start, window_end = _parse_window(execution_row.get("entry_window_et"))
        if window_start is not None:
            execution["entry_window_start_et"] = window_start
            execution["entry_window_end_et"] = window_end
        return {
            "symbol": symbol.upper(),
            "strategy": {
                "key": descriptor.strategy_key,
                "version": 1,
                "params": {
                    **strategy_params,
                    "direction": direction,
                },
            },
            "execution": execution,
            "risk": dict(_PROPOSED_RISK_DEFAULTS),
            "exit": dict(_PROPOSED_EXIT_DEFAULTS),
        }

    def _context_key(self, symbol: str, bias_template: str, horizon: str) -> str:
        return f"{symbol.upper()}|{bias_template}|{horizon}"

    def _candidate_id(self, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
        return f"{payload['strategy_key']}_{payload['symbol'].lower()}_{payload['direction']}_{digest}"

    def _deployment_id(self, strategy_key: str, symbol: str, direction: str, candidate_id: str) -> str:
        return f"{strategy_key}_{symbol.lower()}_{direction}_shadow_{candidate_id[-8:]}"

    def _matching_summary_row(self, m4_summary: pl.DataFrame, row: dict[str, Any]) -> dict[str, Any]:
        filtered = self._filter_matching_rows(m4_summary, row, _M4_SUMMARY_METRIC_FIELDS)
        if filtered.is_empty():
            return {}
        return _jsonable_row(filtered.row(0, named=True))

    def _matching_detail_rows(self, m4_detail: pl.DataFrame, row: dict[str, Any]) -> list[dict[str, Any]]:
        filtered = self._filter_matching_rows(m4_detail, row, set())
        return [_jsonable_row(detail_row) for detail_row in filtered.iter_rows(named=True)]

    def _filter_matching_rows(self, frame: pl.DataFrame, row: dict[str, Any], metric_fields: set[str]) -> pl.DataFrame:
        filtered = frame
        for column, value in row.items():
            if column in _M5_METRIC_FIELDS or column in metric_fields or column not in frame.columns:
                continue
            filtered = filtered.filter(pl.col(column) == value)
        return filtered

    def _stage_entry(
        self,
        research_manifest: dict[str, Any],
        stage: str,
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        for entry in research_manifest.get("stages", []):
            if entry.get("stage") == stage:
                return entry
        if required:
            raise ValueError(f"Run manifest is missing stage {stage}")
        return None

    def _load_csv(self, run_dir: Path, stage_entry: dict[str, Any], artifact_name: str) -> pl.DataFrame:
        artifact_path = stage_entry.get("artifacts", {}).get(artifact_name)
        if not artifact_path:
            return pl.DataFrame()
        absolute = (self.project_root / artifact_path).resolve()
        if not absolute.exists():
            fallback = (run_dir / Path(artifact_path).name).resolve()
            if fallback.exists():
                absolute = fallback
            else:
                raise FileNotFoundError(f"Missing artifact {artifact_name} for stage {stage_entry.get('stage')}: {artifact_path}")
        try:
            return pl.read_csv(absolute)
        except NoDataError:
            return pl.DataFrame()

    def _artifact_path(self, stage_entry: dict[str, Any], artifact_name: str) -> str | None:
        artifact_path = stage_entry.get("artifacts", {}).get(artifact_name)
        return str(artifact_path) if artifact_path is not None else None

    def _relpath(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())

    def _resolve_watchlist(self, candidates: list[ExportCandidate], explicit_watchlist: list[str] | None) -> list[str]:
        if explicit_watchlist:
            return sorted({symbol.upper() for symbol in explicit_watchlist})
        return sorted({candidate.symbol.upper() for candidate in candidates})

    def _resolve_enabled_strategy_families(
        self,
        candidates: list[ExportCandidate],
        explicit_families: list[str] | None,
    ) -> list[str]:
        if explicit_families:
            return sorted(set(explicit_families))
        return sorted({candidate.strategy_key for candidate in candidates})


@dataclass(slots=True, frozen=True)
class _StrategyDescriptor:
    strategy_key: str


def _strategy_descriptor(strategy_name: str) -> _StrategyDescriptor:
    if strategy_name.startswith("Market Impulse"):
        return _StrategyDescriptor(strategy_key="market_impulse")
    if strategy_name.startswith("Jerk-Pivot Momentum"):
        return _StrategyDescriptor(strategy_key="jerk_pivot_momentum")
    if strategy_name.startswith("Elastic Band"):
        return _StrategyDescriptor(strategy_key="elastic_band_reversion")
    if strategy_name.startswith("Opening Drive"):
        return _StrategyDescriptor(strategy_key="opening_drive_classifier")
    raise ValueError(f"Unsupported strategy for loop export: {strategy_name}")


def _bias_template(
    strategy_key: str,
    direction: str,
    *,
    strategy_params: dict[str, Any] | None = None,
) -> str:
    if strategy_key == "opening_drive_classifier":
        return _opening_drive_bias_template(direction, strategy_params or {})
    try:
        return _BIAS_CONTEXTS[(strategy_key, direction)]
    except KeyError as exc:
        raise ValueError(f"No bias template mapping for {strategy_key} {direction}") from exc


def _opening_drive_bias_template(direction: str, strategy_params: dict[str, Any]) -> str:
    enable_continue = bool(strategy_params.get("enable_continue", True))
    enable_fail = bool(strategy_params.get("enable_fail", True))
    if enable_continue and not enable_fail:
        return "bullish_trend_intraday" if direction == "long" else "bearish_trend_intraday"
    if enable_fail and not enable_continue:
        return "bullish_mean_reversion_intraday" if direction == "long" else "bearish_mean_reversion_intraday"
    # Mixed-mode Opening Drive deployments can express both continuation and failure.
    # Until we split those into explicit playbook variants, default them to the
    # continuation lane so they still export into a live-capable surface.
    return "bullish_trend_intraday" if direction == "long" else "bearish_trend_intraday"


def _stage_run_date(stage_entry: dict[str, Any]) -> str:
    recorded_at = str(stage_entry.get("recorded_at", ""))
    if recorded_at:
        return recorded_at[:10]
    return datetime.now(UTC).date().isoformat()


def _ranking_score(row: dict[str, Any], m4_summary: pl.DataFrame) -> float:
    summaries = _matching_summary_rows(m4_summary, row)
    summary_row = _jsonable_row(summaries[0]) if summaries else {}
    return round(
        (_maybe_float(row.get("mc_prob_positive_exp")) or 0.0) * 1000.0
        + (_maybe_float(row.get("mc_exp_r_p50")) or 0.0) * 100.0
        + (_maybe_float(row.get("base_exp_r")) or 0.0) * 10.0
        + (_maybe_float(summary_row.get("mean_holdout_exp_r")) or 0.0),
        6,
    )


def _matching_summary_rows(frame: pl.DataFrame, row: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = frame
    for column, value in row.items():
        if column in _M5_METRIC_FIELDS or column in _M4_SUMMARY_METRIC_FIELDS or column not in frame.columns:
            continue
        filtered = filtered.filter(pl.col(column) == value)
    return list(filtered.iter_rows(named=True))


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in row.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_range(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return None, None
    if raw.lower() in {"n/a", "na", "none", "null"}:
        return None, None
    if "-" in raw:
        start_raw, end_raw = (part.strip() for part in raw.split("-", 1))
        try:
            return int(float(start_raw)), int(float(end_raw))
        except (TypeError, ValueError):
            return None, None
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        return None, None
    return parsed, parsed


def _parse_window(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw or "-" not in raw:
        return None, None
    start_raw, end_raw = (part.strip() for part in raw.split("-", 1))
    return start_raw, end_raw


def _coverage_status(
    *,
    supported: list[ExportCandidate],
    proposed: list[ExportCandidate],
    covered_by_families: list[str],
) -> str:
    if supported or proposed:
        return "researched_with_survivors"
    if covered_by_families:
        return "researched_no_survivors"
    return "not_covered_by_enabled_family"


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


__all__ = ["LoopArtifactExporter", "ExportCandidate"]
