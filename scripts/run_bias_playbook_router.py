#!/usr/bin/env python3
"""Route structured bias inputs into one top playbook per symbol/bias context."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, settings
from src.research import (
    publish_armed_playbooks_to_bhiksha,
    route_bias_inputs,
    route_google_sheet_bias_inputs,
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
DEFAULT_BHIKSHA_ROOT = settings.bhiksha_root.strip() or "../bhiksha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bias-inputs",
        default=None,
        help="Path to CSV export of the human bias sheet.",
    )
    parser.add_argument(
        "--google-sheet-id",
        default=None,
        help="Spreadsheet id or full URL. Defaults to BIONIC_SHEET_ID from .env.",
    )
    parser.add_argument(
        "--google-sheet-name",
        default=DEFAULT_BIONIC_SHEET_NAME,
        help=f"Worksheet name inside the spreadsheet. Defaults to {DEFAULT_BIONIC_SHEET_NAME}.",
    )
    parser.add_argument(
        "--google-credentials",
        default=_default_credentials_path(),
        help="Path to Google service-account credentials JSON.",
    )
    parser.add_argument(
        "--no-sheet-update",
        action="store_true",
        help="Read from Google Sheets but do not write Translator_* fields back.",
    )
    parser.add_argument(
        "--publish-bhiksha",
        action="store_true",
        help="Copy the generated armed manifests into Bhiksha config/deployments/generated after routing.",
    )
    parser.add_argument(
        "--bhiksha-root",
        default=DEFAULT_BHIKSHA_ROOT,
        help=f"Path to the Bhiksha repo root. Defaults to {DEFAULT_BHIKSHA_ROOT}.",
    )
    parser.add_argument(
        "--playbook-catalog",
        required=True,
        help="Path to a nightly playbook_catalog.json bundle.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where routing report, armed payloads, and Bhiksha manifests will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bias_inputs and not (args.google_sheet_id or DEFAULT_BIONIC_SHEET_ID):
        raise SystemExit("Google Sheet id is required; set --google-sheet-id or BIONIC_SHEET_ID in .env")
    if not args.bias_inputs and not args.google_credentials:
        raise SystemExit("Google credentials path is required; set --google-credentials or GOOGLE_API_CREDENTIALS_PATH in .env")
    if args.bias_inputs:
        routing_report_path, armed_payloads_path, selections = route_bias_inputs(
            bias_inputs_path=args.bias_inputs,
            playbook_catalog_path=args.playbook_catalog,
            out_dir=args.out_dir,
        )
    else:
        routing_report_path, armed_payloads_path, selections = route_google_sheet_bias_inputs(
            spreadsheet_id=args.google_sheet_id or DEFAULT_BIONIC_SHEET_ID,
            sheet_name=args.google_sheet_name,
            credentials_path=args.google_credentials,
            playbook_catalog_path=args.playbook_catalog,
            out_dir=args.out_dir,
            update_sheet=not args.no_sheet_update,
        )
    armed_count = sum(1 for selection in selections if selection.status == "armed")
    no_match_count = sum(1 for selection in selections if selection.status != "armed")
    print(f"BIAS_ROUTING_REPORT={routing_report_path}")
    print(f"ARMED_PLAYBOOKS={armed_payloads_path}")
    print(f"ARMED_COUNT={armed_count}")
    print(f"UNARMED_COUNT={no_match_count}")
    if args.publish_bhiksha:
        report = publish_armed_playbooks_to_bhiksha(
            armed_playbooks_path=armed_payloads_path,
            bhiksha_root=args.bhiksha_root,
        )
        print(f"BHIKSHA_GENERATED_DIR={report.generated_dir}")
        print(f"BHIKSHA_PUBLISHED_COUNT={report.published_count}")
        print(f"BHIKSHA_IMPORT_REPORT={report.import_report_path}")


if __name__ == "__main__":
    main()
