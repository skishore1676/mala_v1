#!/usr/bin/env python3
"""Seed or refresh the master playbook catalog from historical M5 survivors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import seed_master_playbook_catalog_from_queue
from src.research import seed_master_playbook_catalog_from_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", help="Path to m2_human_review_queue.csv")
    parser.add_argument("--source-playbook-catalog", help="Optional existing playbook_catalog.json to seed from")
    parser.add_argument("--catalog-path", default=None, help="Optional master catalog JSON path override")
    parser.add_argument("--projection-path", default=None, help="Optional master catalog CSV path override")
    parser.add_argument("--no-sheet-sync", action="store_true", help="Skip syncing the Master_Playbook_Catalog sheet tab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.queue_path and not args.source_playbook_catalog:
        raise SystemExit("Provide --queue-path or --source-playbook-catalog")
    if args.source_playbook_catalog:
        catalog_path, projection_path, playbooks = seed_master_playbook_catalog_from_catalog(
            source_catalog_path=Path(args.source_playbook_catalog),
            catalog_path=Path(args.catalog_path) if args.catalog_path else None,
            projection_path=Path(args.projection_path) if args.projection_path else None,
            sync_sheet=not args.no_sheet_sync,
        )
    else:
        catalog_path, projection_path, playbooks = seed_master_playbook_catalog_from_queue(
            queue_path=Path(args.queue_path),
            catalog_path=Path(args.catalog_path) if args.catalog_path else None,
            projection_path=Path(args.projection_path) if args.projection_path else None,
            sync_sheet=not args.no_sheet_sync,
        )
    print(f"MASTER_PLAYBOOK_CATALOG={catalog_path}")
    if projection_path is not None:
        print(f"MASTER_PLAYBOOK_PROJECTION={projection_path}")
    print(f"PLAYBOOK_COUNT={len(playbooks)}")
    print(
        "STATUS_COUNTS="
        + ",".join(
            f"{status}:{sum(1 for playbook in playbooks if playbook.lifecycle_status == status)}"
            for status in ("active", "stale", "retired")
        )
    )


if __name__ == "__main__":
    main()
