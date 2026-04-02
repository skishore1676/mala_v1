#!/usr/bin/env python3
"""Replay selected human-approved follow-up actions from the durable review queue."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import HumanReviewQueueManager, load_nightly_regime_matrix_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/nightly_regime_matrix.yaml",
        help="Path to nightly regime matrix YAML config.",
    )
    parser.add_argument(
        "--candidate-key",
        action="append",
        dest="candidate_keys",
        default=None,
        help="Candidate key to replay. May be passed multiple times.",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Replay only rows whose human_notes still contain a recorded follow-up error.",
    )
    parser.add_argument(
        "--run-date",
        default=date.today().isoformat(),
        help="Run date to stamp on replayed actions (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_nightly_regime_matrix_config(args.config)
    manager = HumanReviewQueueManager(config.research_control_root)
    run_date = date.fromisoformat(args.run_date)
    rows, replayed = manager.replay_followups(
        config=config,
        run_date=run_date,
        candidate_keys=args.candidate_keys,
        only_error_rows=args.errors_only,
    )
    print(f"REPLAYED_COUNT={replayed}")
    print(f"QUEUE_PATH={manager.paths.queue_path}")
    print(f"WORKBOOK_PATH={manager.paths.workbook_path}")
    print(f"ROW_COUNT={len(rows)}")


if __name__ == "__main__":
    main()
