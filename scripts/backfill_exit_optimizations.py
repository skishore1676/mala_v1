#!/usr/bin/env python3
"""Backfill optimized thesis exits for older M5 survivors."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.research.exit_backfill import backfill_exit_optimizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", required=True, help="Path to m2_human_review_queue.csv")
    parser.add_argument("--start", required=True, help="Research slice start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Research slice end date (YYYY-MM-DD)")
    parser.add_argument("--holdout-start", required=True, help="Holdout start date (YYYY-MM-DD)")
    parser.add_argument("--holdout-end", required=True, help="Holdout end date (YYYY-MM-DD)")
    parser.add_argument("--playbook-catalog", help="Optional playbook_catalog.json to refresh after backfill")
    parser.add_argument("--playbook-projection", help="Optional playbook_catalog.csv projection path")
    parser.add_argument("--force", action="store_true", help="Recompute even if m5_exit_optimization.json already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = backfill_exit_optimizations(
        queue_path=Path(args.queue_path),
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        holdout_start=_parse_date(args.holdout_start),
        holdout_end=_parse_date(args.holdout_end),
        playbook_catalog_path=Path(args.playbook_catalog) if args.playbook_catalog else None,
        playbook_projection_path=Path(args.playbook_projection) if args.playbook_projection else None,
        force=args.force,
    )
    print(f"QUEUE_PATH={result.queue_path}")
    print(f"PROCESSED={result.processed}")
    print(f"OPTIMIZED={result.optimized}")
    print(f"SKIPPED_EXISTING={result.skipped_existing}")
    print(f"SKIPPED_INELIGIBLE={result.skipped_ineligible}")
    if result.playbook_catalog_path is not None:
        print(f"PLAYBOOK_CATALOG={result.playbook_catalog_path}")


def _parse_date(raw: str):
    from datetime import date

    return date.fromisoformat(raw)


if __name__ == "__main__":
    main()
