#!/usr/bin/env python3
"""Publish today's market regime to state/regime-{date}.json for trade_lab.

Run on the laptop as part of the morning sequence — before the gSheet
bridge push, before trade_lab's orchestrator_compute runs on Sunny.

The output file is the authoritative regime source for
``trade_lab/scripts/orchestrator_compute.py``. Its schema is the minimal
tuple the orchestrator reads via ``_resolve_regime``:

    {
      "trading_date": "2026-04-13",
      "vix_band":     "low" | "mid" | "high",
      "spy_trend_20d":"up"  | "flat"| "down",
      "session_type": "normal" | "opex" | "post_fed" | "earnings_heavy",
      "vix_close":    17.34,        # optional, for debugging
      "spy_close":    538.21,       # optional
      "spy_sma20":    531.05,       # optional
      "spy_trend_slope_pct": 0.12,  # optional
      "classified_at": "2026-04-13T06:02:11-04:00",
      "source": "mala_v1.market_regime.classify"
    }

Usage
-----
    python3 scripts/publish_regime_snapshot.py                  # today, default output
    python3 scripts/publish_regime_snapshot.py --date 2026-04-13
    python3 scripts/publish_regime_snapshot.py --out /tmp/regime.json
    python3 scripts/publish_regime_snapshot.py --trade-lab ~/kg_env/projects/trade_lab

The default output path is
``$TRADE_LAB/state/regime-{date}.json`` where ``$TRADE_LAB`` resolves
to ``--trade-lab`` if given, else the ``TRADE_LAB`` environment variable,
else ``~/kg_env/projects/trade_lab``.

Exit codes
----------
    0  file written successfully
    1  polygon / network error
    2  trade_lab state dir not found
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Make `src.research.market_regime` importable when run from the scripts/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.market_regime import classify, VixUnavailable  # noqa: E402


def _default_trade_lab_root() -> Path:
    env = os.environ.get("TRADE_LAB", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / "kg_env" / "projects" / "trade_lab"


def _iso_now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_payload(target_date: dt.date) -> dict:
    """Classify `target_date` and return the JSON-serialisable payload."""
    regime = classify(target_date)
    return {
        "trading_date": regime.trading_date.isoformat(),
        "vix_band": regime.vix_band,
        "spy_trend_20d": regime.spy_trend_20d,
        "session_type": regime.session_type,
        "vix_close": regime.vix_close,
        "spy_close": regime.spy_close,
        "spy_sma20": regime.spy_sma20,
        "spy_trend_slope_pct": regime.spy_trend_slope_pct,
        "classified_at": _iso_now_local(),
        "source": "mala_v1.market_regime.classify",
    }


def resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out).expanduser()
    root = Path(args.trade_lab).expanduser() if args.trade_lab else _default_trade_lab_root()
    state_dir = root / "state"
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"trade_lab state dir not found: {state_dir}. "
            "Pass --trade-lab or --out to override."
        )
    return state_dir / f"regime-{args.date.isoformat()}.json"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--date",
        type=lambda s: dt.date.fromisoformat(s),
        default=dt.date.today(),
        help="Target trading date (default: today). If not a trading day, "
             "the regime of the most recent prior trading day is used.",
    )
    ap.add_argument("--trade-lab", type=str, default=None,
                    help="trade_lab project root (default: $TRADE_LAB or ~/kg_env/projects/trade_lab)")
    ap.add_argument("--out", type=str, default=None,
                    help="Explicit output path; overrides --trade-lab derivation.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the payload to stdout, don't write the file.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        payload = build_payload(args.date)
    except VixUnavailable as e:
        print(f"ERROR: VIX data unavailable — {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: regime classification failed: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    try:
        out_path = resolve_out_path(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if args.verbose:
        print(f"wrote {out_path}")
        print(
            f"  vix_band={payload['vix_band']} "
            f"spy_trend_20d={payload['spy_trend_20d']} "
            f"session_type={payload['session_type']}"
        )
    else:
        print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
