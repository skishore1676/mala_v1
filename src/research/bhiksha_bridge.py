"""Publish armed Mala playbooks into Bhiksha generated deployments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.config import PROJECT_ROOT


@dataclass(slots=True, frozen=True)
class BhikshaPublishReport:
    published_count: int
    generated_dir: str
    manifest_paths: list[str]
    import_report_path: str


def publish_armed_playbooks_to_bhiksha(
    *,
    armed_playbooks_path: str | Path,
    bhiksha_root: str | Path,
) -> BhikshaPublishReport:
    armed_path = Path(armed_playbooks_path).resolve()
    payload = json.loads(armed_path.read_text(encoding="utf-8"))
    manifests = payload.get("armed_playbooks", [])

    root = Path(bhiksha_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    generated_dir = root / "config" / "deployments" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(generated_dir.glob("*.yaml")):
        file_path.unlink()

    published_paths: list[str] = []
    for item in manifests:
        manifest_path = Path(str(item["manifest_path"]))
        if not manifest_path.is_absolute():
            manifest_path = (PROJECT_ROOT / manifest_path).resolve()
        target_path = generated_dir / manifest_path.name
        shutil.copy2(manifest_path, target_path)
        _validate_manifest_shape(target_path)
        published_paths.append(str(target_path))

    import_report = {
        "armed_playbooks_path": str(armed_path),
        "bhiksha_root": str(root),
        "generated_dir": str(generated_dir),
        "published_count": len(published_paths),
        "published_manifests": published_paths,
    }
    import_report_path = root / "artifacts" / "playbook" / "mala_router_publish_report.json"
    import_report_path.parent.mkdir(parents=True, exist_ok=True)
    import_report_path.write_text(json.dumps(import_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BhikshaPublishReport(
        published_count=len(published_paths),
        generated_dir=str(generated_dir),
        manifest_paths=published_paths,
        import_report_path=str(import_report_path),
    )


def _validate_manifest_shape(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest object at {path}")
    required = {"deployment_id", "symbol", "strategy", "execution", "risk", "exit"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Manifest {path} is missing required fields: {missing}")


__all__ = [
    "BhikshaPublishReport",
    "publish_armed_playbooks_to_bhiksha",
]
