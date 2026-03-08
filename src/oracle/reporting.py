"""
Oracle Experiment Reporter

Saves comprehensive experiment results to timestamped directories:

    data/results/
    └── 2026-02-12_18-56_SPY_EMA-Momentum/
        ├── experiment.json   ← full config, metrics, and metadata
        ├── trade_log.csv     ← every signal with forward metrics
        └── summary.txt       ← human-readable report

Each run is immutable — nothing is overwritten.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from loguru import logger

from src.config import DATA_DIR


class ExperimentReporter:
    """Persist experiment results to a timestamped directory."""

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        self.results_dir = results_dir or DATA_DIR / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(
        self,
        *,
        ticker: str,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        date_range: tuple[date, date],
        total_bars: int,
        enriched_columns: List[str],
        summary_df: pl.DataFrame,
        trade_log_df: pl.DataFrame,
        physics_params: Dict[str, Any],
        oracle_params: Dict[str, Any],
    ) -> Path:
        """
        Save a complete experiment snapshot.

        Returns the path to the experiment directory.
        """
        # ── Create timestamped directory ─────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = strategy_name.replace(" ", "-").replace("/", "-")
        dir_name = f"{ts}_{ticker}_{safe_name}"
        exp_dir = self.results_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ── Build the experiment manifest ────────────────────────────────
        summary_dict = self._df_to_dict(summary_df)

        experiment = {
            "metadata": {
                "timestamp": ts,
                "ticker": ticker,
                "date_range": {
                    "start": date_range[0].isoformat(),
                    "end": date_range[1].isoformat(),
                },
                "total_bars": total_bars,
                "enriched_columns": enriched_columns,
            },
            "strategy": {
                "name": strategy_name,
                "params": strategy_params,
            },
            "physics": physics_params,
            "oracle": oracle_params,
            "results": summary_dict,
        }

        # ── Write experiment.json ────────────────────────────────────────
        json_path = exp_dir / "experiment.json"
        with open(json_path, "w") as f:
            json.dump(experiment, f, indent=2, default=str)
        logger.info("Experiment manifest → {}", json_path)

        # ── Write trade_log.csv ──────────────────────────────────────────
        csv_path = exp_dir / "trade_log.csv"
        trade_log_df.write_csv(csv_path)
        logger.info("Trade log ({} rows) → {}", len(trade_log_df), csv_path)

        # ── Write human-readable summary.txt ─────────────────────────────
        txt_path = exp_dir / "summary.txt"
        self._write_summary_txt(txt_path, experiment, len(trade_log_df))
        logger.info("Summary report → {}", txt_path)

        return exp_dir

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiments with key metrics."""
        experiments = []
        for exp_dir in sorted(self.results_dir.iterdir()):
            json_path = exp_dir / "experiment.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                experiments.append({
                    "dir": str(exp_dir),
                    "timestamp": data["metadata"]["timestamp"],
                    "ticker": data["metadata"]["ticker"],
                    "strategy": data["strategy"]["name"],
                    "total_signals": data["results"].get("total_signals"),
                    "confidence_score": data["results"].get("confidence_score"),
                })
        return experiments

    # ── Private Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _df_to_dict(df: pl.DataFrame) -> Dict[str, Any]:
        """Convert a single-row summary DataFrame to a dict."""
        if df.is_empty():
            return {}
        row = df.row(0, named=True)
        # Convert numpy/polars types to native Python for JSON
        return {k: _to_native(v) for k, v in row.items()}

    @staticmethod
    def _write_summary_txt(
        path: Path,
        experiment: Dict[str, Any],
        trade_count: int,
    ) -> None:
        """Write a human-readable summary report."""
        m = experiment["metadata"]
        s = experiment["strategy"]
        r = experiment["results"]

        lines = [
            "=" * 60,
            "  KINEMATIC ENGINE – EXPERIMENT REPORT",
            "=" * 60,
            "",
            f"  Timestamp:   {m['timestamp']}",
            f"  Ticker:      {m['ticker']}",
            f"  Date Range:  {m['date_range']['start']} → {m['date_range']['end']}",
            f"  Total Bars:  {m['total_bars']:,}",
            "",
            "-" * 60,
            "  STRATEGY",
            "-" * 60,
            f"  Name:   {s['name']}",
            f"  Params: {json.dumps(s['params'], indent=10)}",
            "",
            "-" * 60,
            "  PHYSICS CONFIG",
            "-" * 60,
            f"  {json.dumps(experiment['physics'], indent=2)}",
            "",
            "-" * 60,
            "  ORACLE CONFIG",
            "-" * 60,
            f"  Forward Window: {experiment['oracle'].get('forward_window_bars', 'N/A')} bars",
            "",
            "-" * 60,
            "  RESULTS",
            "-" * 60,
        ]

        if r:
            max_key_len = max(len(k) for k in r)
            for key, val in r.items():
                label = key.replace("_", " ").title()
                if isinstance(val, float):
                    if "confidence" in key:
                        display = f"{val:.2%}"
                    else:
                        display = f"{val:.4f}"
                else:
                    display = f"{val:,}" if isinstance(val, int) else str(val)
                lines.append(f"  {label:<{max_key_len + 5}} {display}")
        else:
            lines.append("  No valid signals found.")

        lines += [
            "",
            f"  Trade Log:   {trade_count:,} entries",
            "",
            "=" * 60,
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def _to_native(val: Any) -> Any:
    """Convert numpy/polars scalars to native Python types."""
    import numpy as np

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return v if not np.isnan(v) else None
    if isinstance(val, float):
        import math
        return val if not math.isnan(val) else None
    return val
