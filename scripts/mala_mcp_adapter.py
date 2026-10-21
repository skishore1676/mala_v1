#!/usr/bin/env python3
"""
Mala MCP Adapter — called by mcp_backtest.py via subprocess.

Usage:
    /path/to/mala_v1/.venv/bin/python scripts/mala_mcp_adapter.py <tool_name> '<json_kwargs>'

Outputs JSON to stdout. All errors go to stderr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is importable when run from any working directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))  # adds project root so `from src.x import y` works

from src.research.orchestrator import ResearchOrchestrator
from src.research.models import ResearchStage


_STAGE_MAP: dict[str, ResearchStage] = {
    "M1": ResearchStage.M1_DISCOVERY,
    "M2": ResearchStage.M2_CONVERGENCE,
    "M3": ResearchStage.M3_WALK_FORWARD,
    "M4": ResearchStage.M4_HOLDOUT,
    "M5": ResearchStage.M5_EXECUTION,
}

_TOOL_STAGE: dict[str, ResearchStage] = {
    "parameter_sweep":      ResearchStage.M1_DISCOVERY,
    "baseline_comparison":  ResearchStage.M1_DISCOVERY,
    "evaluate_config":      ResearchStage.M1_DISCOVERY,
    "query_incumbent":      ResearchStage.M1_DISCOVERY,
    "query_pareto_front":   ResearchStage.M1_DISCOVERY,
    "query_dead_zones":     ResearchStage.M1_DISCOVERY,
    "convergence_grid":     ResearchStage.M2_CONVERGENCE,
    "ablation_check":       ResearchStage.M2_CONVERGENCE,
    "query_neighborhood":   ResearchStage.M2_CONVERGENCE,
    "walk_forward":         ResearchStage.M3_WALK_FORWARD,
    "holdout_validation":   ResearchStage.M4_HOLDOUT,
    "execution_mapping":    ResearchStage.M5_EXECUTION,
}

_VALIDATION_STRATEGIES = [
    "Elastic Band Reversion",
    "Jerk-Pivot Momentum (tight)",
    "Opening Drive Classifier",
    "Market Impulse (Cross & Reclaim)",
]


def _json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def cmd_list_strategies(_kwargs: dict) -> dict:
    orch = ResearchOrchestrator()
    entries = orch.validation_entries()
    return {
        "strategies": [
            {
                "name": e.name,
                "status": e.status.value if hasattr(e.status, "value") else str(e.status),
                "tickers": e.tickers,
                "directions": e.directions,
                "required_features": sorted(e.required_features),
                "parameter_space": _json_safe(e.parameter_space),
                "strategy_config": _json_safe(e.strategy_config),
                "notes": e.notes,
            }
            for e in entries
        ],
        "validation_set": _VALIDATION_STRATEGIES,
    }


def _lists_to_tuples(value):
    """
    Recursively convert lists-of-scalars to tuples so they match Mala's
    internal representation (e.g. vwma_periods: [8,21,34] → (8,21,34)).
    Lists-of-dicts or lists-of-strings are left as lists.
    """
    if isinstance(value, list):
        if all(isinstance(v, (int, float)) for v in value):
            return tuple(value)
        return [_lists_to_tuples(v) for v in value]
    return value


def _coerce_kwargs(tool_name: str, kwargs: dict) -> dict:
    """
    Type-coerce kwargs before passing to ResearchToolbox.
    - start_date / end_date: ISO string → datetime.date
    - tickers: ensure list of str
    - config values: nested lists of numbers → tuples (for vwma_periods etc.)
    """
    from datetime import date as _date

    out = dict(kwargs)

    for field in ("start_date", "end_date"):
        val = out.get(field)
        if isinstance(val, str):
            try:
                out[field] = _date.fromisoformat(val)
            except ValueError:
                pass

    if "tickers" in out and not isinstance(out["tickers"], list):
        out["tickers"] = [str(out["tickers"])]

    # Coerce config dict values: nested lists of numbers → tuples
    if "config" in out and isinstance(out["config"], dict):
        out["config"] = {k: _lists_to_tuples(v) for k, v in out["config"].items()}

    return out


def cmd_run_tool(tool_name: str, kwargs: dict) -> dict:
    if tool_name not in _TOOL_STAGE:
        return {"error": f"Unknown tool: {tool_name!r}. Valid: {sorted(_TOOL_STAGE)}"}

    stage = _TOOL_STAGE[tool_name]

    # Override stage if caller specified one explicitly
    if "stage" in kwargs:
        stage_key = str(kwargs.pop("stage")).upper()
        stage = _STAGE_MAP.get(stage_key, stage)

    kwargs = _coerce_kwargs(tool_name, kwargs)

    orch = ResearchOrchestrator()
    try:
        result = orch.run_action(stage, tool_name, **kwargs)
        return {
            "tool_name": result.tool_name,
            "summary": _json_safe(result.summary),
            "artifacts": _json_safe(result.artifacts),
        }
    except Exception as exc:
        return {"error": str(exc), "tool_name": tool_name}


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: mala_mcp_adapter.py <tool_name> [json_kwargs]"}))
        sys.exit(1)

    tool_name = sys.argv[1]
    raw_kwargs = sys.argv[2] if len(sys.argv) > 2 else "{}"

    try:
        kwargs = json.loads(raw_kwargs)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON kwargs: {exc}"}))
        sys.exit(1)

    if tool_name == "list_strategies":
        result = cmd_list_strategies(kwargs)
    else:
        result = cmd_run_tool(tool_name, kwargs)

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
