#!/usr/bin/env python3
"""Push regime + catalog_regime_performance blobs to the trade_lab_bridge sheet tab.

Run on the laptop after publish_regime_snapshot.py and after the nightly
M3-M5 runner updates catalog_regime_performance.json.

The Master_Playbook_Catalog tab is already synced by
``src.research.playbooks.sync_master_playbook_catalog_sheet()`` — this
script does NOT touch that tab.

What this pushes to the ``trade_lab_bridge`` tab:

    filename                               | payload_json       | written_at
    catalog_regime_performance.json        | {"rows": [...]}    | 2026-04-13T...
    regime-YYYY-MM-DD.json                 | {"vix_band":...}   | 2026-04-13T...

Usage
-----
    python3 scripts/push_trade_lab_bridge.py
    python3 scripts/push_trade_lab_bridge.py --date 2026-04-13
    python3 scripts/push_trade_lab_bridge.py --regime-only   # skip perf push
    python3 scripts/push_trade_lab_bridge.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings, PROJECT_ROOT  # noqa: E402
from src.research.google_sheets import GoogleSheetTableClient  # noqa: E402


def _iso_now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _default_trade_lab_root() -> Path:
    import os
    env = os.environ.get("TRADE_LAB", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / "kg_env" / "projects" / "trade_lab"


def _resolve_credentials_path() -> Path:
    raw = settings.google_api_credentials_path.strip()
    if not raw:
        raise RuntimeError("GOOGLE_API_CREDENTIALS_PATH not configured in .env")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _build_client(sheet_name: str) -> GoogleSheetTableClient:
    import os
    sheet_id = (
        os.environ.get("TRADE_LAB_BRIDGE_SHEET_ID", "").strip()
        or settings.bionic_sheet_id.strip()
    )
    if not sheet_id:
        raise RuntimeError("No sheet ID available — set TRADE_LAB_BRIDGE_SHEET_ID or BIONIC_SHEET_ID")
    return GoogleSheetTableClient(
        spreadsheet_id=sheet_id,
        sheet_name=sheet_name,
        credentials_path=_resolve_credentials_path(),
    )


def push_blob(client: GoogleSheetTableClient, filename: str, payload_json: str, dry_run: bool = False) -> None:
    """Push one file blob to the bridge tab (upsert by filename)."""
    now = _iso_now_local()
    sha = _sha256(payload_json)

    existing = client.read_rows()
    target_row = None
    for row in existing:
        if row.get("filename") == filename:
            target_row = row
            break

    if target_row is not None:
        target_row["payload_json"] = payload_json
        target_row["written_at"] = now
        target_row["written_by"] = "mala_v1@laptop"
        target_row["sha256"] = sha
        if dry_run:
            print(f"  [dry-run] would update row {target_row['row_index']} for {filename}")
            return
        client.batch_update_rows(
            rows=[target_row],
            columns=["payload_json", "written_at", "written_by", "sha256"],
        )
    else:
        headers = ["filename", "payload_json", "written_at", "written_by", "sha256"]
        new_rows = existing + [{
            "filename": filename,
            "payload_json": payload_json,
            "written_at": now,
            "written_by": "mala_v1@laptop",
            "sha256": sha,
        }]
        if dry_run:
            print(f"  [dry-run] would append new row for {filename}")
            return
        client.overwrite_table(headers=headers, rows=new_rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=lambda s: dt.date.fromisoformat(s), default=dt.date.today())
    ap.add_argument("--regime-only", action="store_true", help="Skip catalog_regime_performance push")
    ap.add_argument("--perf-only", action="store_true", help="Skip regime push")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sheet-name", default=None, help="Override bridge tab name")
    args = ap.parse_args(argv)

    import os
    sheet_name = args.sheet_name or os.environ.get("TRADE_LAB_BRIDGE_SHEET_NAME", "trade_lab_bridge")
    client = _build_client(sheet_name)

    perf_path = PROJECT_ROOT / "data" / "playbooks" / "catalog_regime_performance.json"
    trade_lab_root = _default_trade_lab_root()
    regime_path = trade_lab_root / "state" / f"regime-{args.date.isoformat()}.json"

    pushed = 0

    if not args.regime_only:
        if perf_path.exists():
            payload = perf_path.read_text().strip()
            if args.verbose:
                print(f"pushing catalog_regime_performance.json ({len(payload)} chars)")
            push_blob(client, "catalog_regime_performance.json", payload, dry_run=args.dry_run)
            pushed += 1
        else:
            print(f"WARN: {perf_path} not found — skipping", file=sys.stderr)

    if not args.perf_only:
        regime_filename = f"regime-{args.date.isoformat()}.json"
        if regime_path.exists():
            payload = regime_path.read_text().strip()
            if args.verbose:
                print(f"pushing {regime_filename} ({len(payload)} chars)")
            push_blob(client, regime_filename, payload, dry_run=args.dry_run)
            pushed += 1
        else:
            print(f"WARN: {regime_path} not found — skipping", file=sys.stderr)

    print(f"pushed {pushed} blob(s) to {sheet_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
