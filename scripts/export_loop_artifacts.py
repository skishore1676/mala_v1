#!/usr/bin/env python3
"""Export Mala run directories into Bhiksha loop artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import LoopArtifactExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deployment candidates and playbook catalog from Mala run directories.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        dest="run_dirs",
        help="Run directory containing research_manifest.json. Can be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for deployment_candidates.json and playbook_catalog.json. Defaults to the single run directory if exactly one is provided.",
    )
    parser.add_argument(
        "--watchlist",
        nargs="+",
        default=None,
        help="Optional explicit watchlist for full matrix exports.",
    )
    parser.add_argument(
        "--enabled-strategy-family",
        action="append",
        dest="enabled_strategy_families",
        default=None,
        help="Optional strategy family to mark as researched. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif len(args.run_dirs) == 1:
        out_dir = Path(args.run_dirs[0])
    else:
        out_dir = Path("data/results/loop_exports/latest")
    exporter = LoopArtifactExporter()
    candidates_path, playbook_path = exporter.export_runs(
        args.run_dirs,
        out_dir=out_dir,
        watchlist=args.watchlist,
        enabled_strategy_families=args.enabled_strategy_families,
    )
    print(f"DEPLOYMENT_CANDIDATES={candidates_path}")
    print(f"PLAYBOOK_CATALOG={playbook_path}")


if __name__ == "__main__":
    main()
