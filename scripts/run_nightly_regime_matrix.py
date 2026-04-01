#!/usr/bin/env python3
"""Run the nightly regime matrix across the fixed watchlist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import load_nightly_regime_matrix_config, run_nightly_regime_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/nightly_regime_matrix.yaml",
        help="Path to nightly regime matrix YAML config.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional explicit output bundle directory. Defaults to a dated run under the configured output_root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_nightly_regime_matrix_config(args.config)
    result = run_nightly_regime_matrix(config, bundle_dir=args.bundle_dir)
    print(f"BROAD_SCOUT_MAX_STAGE={config.defaults.broad_scout_max_stage}")
    print(f"scout_only_run = {str(result.scout_only_run).lower()}")
    print(f"deployment_candidates_generated = {result.deployment_candidates_generated}")
    print(f"followup_actions_run_count = {result.followup_actions_run_count}")
    print(f"reason = {result.summary_reason}")
    print(f"BUNDLE_DIR={result.bundle_dir}")
    for family, run_dir in sorted(result.run_dirs.items()):
        print(f"RUN_DIR_{family.upper()}={run_dir}")
    for family, log_path in sorted(result.family_log_paths.items()):
        print(f"RUN_LOG_{family.upper()}={log_path}")
    print(f"DEPLOYMENT_CANDIDATES={result.deployment_candidates_path}")
    print(f"PLAYBOOK_CATALOG={result.playbook_catalog_path}")
    print(f"REVIEW_QUEUE={result.review_queue_path}")
    print(f"REVIEW_HISTORY={result.review_history_path}")
    print(f"REVIEW_WORKBOOK={result.review_workbook_path}")
    print(f"REVIEW_BUNDLE={result.review_bundle_dir}")
    print(f"CHARTS_DIR={result.charts_dir}")
    print(f"NIGHTLY_MATRIX_MANIFEST={result.manifest_path}")


if __name__ == "__main__":
    main()
