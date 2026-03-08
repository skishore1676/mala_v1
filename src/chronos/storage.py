"""
Local Parquet-based storage for the Chronos pipeline.

Organises data as:
    data/<TICKER>/<YYYY-MM-DD>.parquet

Provides:
  - save_bars()    → persist raw API results
  - load_bars()    → read back into a Polars DataFrame
  - missing_dates() → identify gaps for incremental downloads
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl
from loguru import logger

from src.config import DATA_DIR

# Unix-epoch ms → date helpers
_MS_PER_DAY = 86_400_000


class LocalStorage:
    """Read/write 1-min bar Parquet files, one file per ticker per day."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────

    def save_bars(self, ticker: str, raw_bars: List[dict]) -> int:
        """
        Persist raw Polygon bar dicts to per-day Parquet files.
        Returns the number of files written.
        """
        if not raw_bars:
            logger.warning("No bars to save for {}", ticker)
            return 0

        df = self._bars_to_dataframe(raw_bars, ticker)
        # Add a date column for partitioning
        df = df.with_columns(pl.col("timestamp").dt.date().alias("trade_date"))

        written = 0
        for trade_date, group in df.group_by("trade_date"):
            day: date = trade_date[0]  # type: ignore[index]
            path = self._path_for(ticker, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            group.drop("trade_date").write_parquet(path)
            written += 1

        logger.info("Saved {} daily files for {}", written, ticker)
        return written

    def load_bars(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pl.DataFrame:
        """Load and concatenate stored Parquet files for *ticker*."""
        ticker_dir = self.base_dir / ticker.upper()
        if not ticker_dir.exists():
            logger.warning("No data directory found for {}", ticker)
            return pl.DataFrame()

        files = sorted(ticker_dir.glob("*.parquet"))
        if start or end:
            files = [
                f
                for f in files
                if self._file_in_range(f, start, end)
            ]

        if not files:
            return pl.DataFrame()

        frames = [self._normalize_loaded_frame(pl.read_parquet(f)) for f in files]
        df = pl.concat(frames).sort("timestamp")
        logger.info(
            "Loaded {} bars for {} ({} files)",
            len(df),
            ticker,
            len(files),
        )
        return df

    def existing_dates(self, ticker: str) -> Set[date]:
        """Return the set of dates already stored for *ticker*."""
        ticker_dir = self.base_dir / ticker.upper()
        if not ticker_dir.exists():
            return set()
        return {
            date.fromisoformat(f.stem)
            for f in ticker_dir.glob("*.parquet")
        }

    def missing_dates(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> List[date]:
        """Return trading dates in [start, end] NOT yet stored."""
        existing = self.existing_dates(ticker)
        all_dates = {
            start + timedelta(days=i)
            for i in range((end - start).days + 1)
            # skip weekends
            if (start + timedelta(days=i)).weekday() < 5
        }
        return sorted(all_dates - existing)

    # ── Private Helpers ──────────────────────────────────────────────────

    def _path_for(self, ticker: str, day: date) -> Path:
        return self.base_dir / ticker.upper() / f"{day.isoformat()}.parquet"

    @staticmethod
    def _bars_to_dataframe(raw_bars: List[dict], ticker: str) -> pl.DataFrame:
        """Convert Polygon bar dicts to a typed Polars DataFrame."""
        df = pl.DataFrame(raw_bars)
        # Polygon uses 't' for Unix-ms timestamp, 'o/h/l/c' for prices,
        # 'v' for volume, 'vw' for VWAP, 'n' for num transactions
        rename_map: Dict[str, str] = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(rename_map)

        # Convert epoch-ms to datetime
        if "timestamp" in df.columns:
            df = df.with_columns(
                (pl.col("timestamp") * 1_000)
                .cast(pl.Datetime("us", time_zone="UTC"))
                .alias("timestamp")
            )

        # Add ticker column
        df = df.with_columns(pl.lit(ticker.upper()).alias("ticker"))

        # Select and order canonical columns (only those present)
        canonical = [
            "timestamp", "ticker", "open", "high", "low",
            "close", "volume", "vwap", "transactions",
        ]
        present = [c for c in canonical if c in df.columns]
        return df.select(present)

    @staticmethod
    def _normalize_loaded_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Coerce cached frames to a stable schema across cache generations."""
        if "timestamp" in df.columns:
            df = df.with_columns(
                pl.col("timestamp")
                .cast(pl.Int64)
                .cast(pl.Datetime("us", time_zone="UTC"))
                .alias("timestamp")
            )
        return df

    @staticmethod
    def _file_in_range(
        path: Path,
        start: Optional[date],
        end: Optional[date],
    ) -> bool:
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            return False
        if start and file_date < start:
            return False
        if end and file_date > end:
            return False
        return True
