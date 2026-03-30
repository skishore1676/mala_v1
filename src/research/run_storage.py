"""Helpers for keeping dated agentic research runs organized."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: str | Path, strategy_slug: str) -> Path:
    now = datetime.now()
    run_dir = Path(base_dir) / now.strftime("%Y-%m-%d") / strategy_slug / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_strategy_index(run_dir: Path, *, strategy_label: str, headline: str) -> Path:
    strategy_dir = run_dir.parent
    index_path = strategy_dir / "INDEX.md"
    if not index_path.exists():
        index_path.write_text(
            "# Agentic Run Index\n\n| run | strategy | headline |\n|---|---|---|\n",
            encoding="utf-8",
        )
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(f"| `{run_dir.name}` | `{strategy_label}` | {headline} |\n")
    return index_path


__all__ = ["append_strategy_index", "create_run_dir"]
