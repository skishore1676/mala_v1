#!/usr/bin/env python3
"""Compile one active_session.json from Bionic playbooks and manual entry rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, settings
from src.research import (
    compile_active_session_from_google_sheets,
    default_master_playbook_catalog_path,
    publish_active_session_to_bhiksha,
)


def _default_credentials_path() -> str:
    configured = settings.google_api_credentials_path.strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return str(path)
    return ""


DEFAULT_BIONIC_SHEET_ID = settings.bionic_sheet_id.strip()
DEFAULT_BIONIC_SHEET_NAME = settings.bionic_sheet_name.strip() or "Bionic_Loop"
DEFAULT_MANUAL_SHEET_ID = settings.manual_entry_sheet_id.strip()
DEFAULT_MANUAL_SHEET_NAME = settings.manual_entry_sheet_name.strip() or "entry_v1"
DEFAULT_BHIKSHA_ROOT = settings.bhiksha_root.strip() or "../bhiksha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--playbook-catalog",
        default=str(default_master_playbook_catalog_path()),
        help="Path to playbook_catalog.json. Defaults to the master playbook catalog.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for active_session artifacts.")
    parser.add_argument("--bionic-google-sheet-id", default=DEFAULT_BIONIC_SHEET_ID)
    parser.add_argument("--bionic-google-sheet-name", default=DEFAULT_BIONIC_SHEET_NAME)
    parser.add_argument("--manual-google-sheet-id", default=DEFAULT_MANUAL_SHEET_ID)
    parser.add_argument("--manual-google-sheet-name", default=DEFAULT_MANUAL_SHEET_NAME)
    parser.add_argument("--google-credentials", default=_default_credentials_path())
    parser.add_argument("--no-bionic-sheet-update", action="store_true")
    parser.add_argument("--live-authorized", action="store_true", help="Emit session deployments with execution.shadow_only=false.")
    parser.add_argument("--publish-bhiksha", action="store_true")
    parser.add_argument("--bhiksha-root", default=DEFAULT_BHIKSHA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bionic_google_sheet_id:
        raise SystemExit("Bionic Google Sheet id is required; set --bionic-google-sheet-id or BIONIC_SHEET_ID in .env")
    if not args.manual_google_sheet_id:
        raise SystemExit("manual Google Sheet id is required; set --manual-google-sheet-id or MANUAL_ENTRY_SHEET_ID in .env")
    if not args.google_credentials:
        raise SystemExit("Google credentials path is required; set --google-credentials or GOOGLE_API_CREDENTIALS_PATH in .env")

    session_path, report_path, payload = compile_active_session_from_google_sheets(
        bionic_spreadsheet_id=args.bionic_google_sheet_id,
        bionic_sheet_name=args.bionic_google_sheet_name,
        manual_spreadsheet_id=args.manual_google_sheet_id,
        manual_sheet_name=args.manual_google_sheet_name,
        credentials_path=args.google_credentials,
        playbook_catalog_path=args.playbook_catalog,
        out_dir=args.out_dir,
        update_bionic_sheet=not args.no_bionic_sheet_update,
        live_authorized=args.live_authorized,
    )
    print(f"ACTIVE_SESSION={session_path}")
    print(f"ACTIVE_SESSION_REPORT={report_path}")
    print(f"DEPLOYMENT_COUNT={payload['summary']['deployment_count']}")
    print(f"MANUAL_DEPLOYMENT_COUNT={payload['summary']['manual_deployment_count']}")
    print(f"PLAYBOOK_DEPLOYMENT_COUNT={payload['summary']['playbook_deployment_count']}")
    print(f"AUTHORIZATION_MODE={payload['authorization_mode']}")
    if args.publish_bhiksha:
        published = publish_active_session_to_bhiksha(
            session_payload_path=session_path,
            bhiksha_root=args.bhiksha_root,
        )
        print(f"BHIKSHA_ACTIVE_SESSION={published}")


if __name__ == "__main__":
    main()
