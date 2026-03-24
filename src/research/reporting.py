"""Helpers for persisting stage-by-stage research outcomes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.research.models import ResearchStage
from src.research.tools import ResearchToolResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return str(value)


class ResearchJournal:
    """Persist concise stage reports plus a machine-readable run manifest."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "research_manifest.json"

    def record_stage(
        self,
        *,
        stage: ResearchStage,
        result: ResearchToolResult,
        decision: str,
        rationale: str,
        next_action: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        artifact_paths = self._persist_artifacts(stage=stage, result=result)
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage.value,
            "tool_name": result.tool_name,
            "decision": decision,
            "rationale": rationale,
            "next_action": next_action,
            "summary": _jsonable(result.summary),
            "context": _jsonable(context or {}),
            "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        }
        manifest = self._read_manifest()
        manifest.setdefault("stages", []).append(entry)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        self._write_stage_markdown(entry)
        return artifact_paths

    def _persist_artifacts(self, *, stage: ResearchStage, result: ResearchToolResult) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        prefix = f"{stage.value}_{result.tool_name}"
        for name, artifact in result.artifacts.items():
            if isinstance(artifact, pl.DataFrame):
                path = self.out_dir / f"{prefix}_{name}.csv"
                artifact.write_csv(path)
                paths[name] = path
        return paths

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"stages": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_stage_markdown(self, entry: dict[str, Any]) -> None:
        path = self.out_dir / f"{entry['stage']}_{entry['tool_name']}.md"
        lines = [
            f"# {entry['stage']} - {entry['tool_name']}",
            "",
            f"- Decision: `{entry['decision']}`",
            f"- Next action: `{entry['next_action']}`",
            "",
            "## Rationale",
            entry["rationale"],
            "",
            "## Summary",
        ]
        for key, value in entry["summary"].items():
            lines.append(f"- `{key}`: `{value}`")
        if entry["context"]:
            lines.extend(["", "## Context"])
            for key, value in entry["context"].items():
                lines.append(f"- `{key}`: `{value}`")
        if entry["artifacts"]:
            lines.extend(["", "## Artifacts"])
            for name, artifact_path in entry["artifacts"].items():
                lines.append(f"- `{name}`: `{artifact_path}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["ResearchJournal"]
