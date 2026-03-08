"""
Oracle Metrics Calculator

Computes forward-looking metrics for every signal bar:
  - MFE  (Maximum Favorable Excursion)   – highest high in next N bars
  - MAE  (Maximum Adverse Excursion)     – lowest low in next N bars
  - Win  flag                             – MFE > 2× MAE
  - Confidence Score                      – Wins / Total Signals

Also produces a summary report DataFrame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from src.config import settings


class MetricsCalculator:
    """Compute MFE / MAE and build probabilistic output surfaces."""

    def __init__(
        self,
        forward_window: int = settings.forward_window_bars,
    ) -> None:
        self.forward_window = forward_window

    # ── Public ───────────────────────────────────────────────────────────

    def add_forward_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Append forward-looking MFE, MAE, and win flag columns.

        Requires 'high', 'low', 'close' columns in the DataFrame.
        Must be called BEFORE filtering to signals-only so that we
        have the full bar window for look-ahead.
        """
        n = len(df)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()

        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)

        for i in range(n - self.forward_window):
            future_high = high[i + 1 : i + 1 + self.forward_window]
            future_low = low[i + 1 : i + 1 + self.forward_window]
            entry_price = close[i]

            mfe[i] = float(np.max(future_high) - entry_price)
            mae[i] = float(entry_price - np.min(future_low))

        df = df.with_columns([
            pl.Series(f"forward_mfe_{self.forward_window}", mfe),
            pl.Series(f"forward_mae_{self.forward_window}", mae),
        ])

        # Win = MFE > 2× MAE (reward/risk > 2)
        mfe_col = f"forward_mfe_{self.forward_window}"
        mae_col = f"forward_mae_{self.forward_window}"
        df = df.with_columns(
            (pl.col(mfe_col) > 2.0 * pl.col(mae_col)).alias("win")
        )

        logger.info("Forward metrics added (window = {} bars)", self.forward_window)
        return df

    def summarise_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Given a DataFrame with 'signal' and forward metrics, produce a
        summary row per ticker.
        """
        if "signal" not in df.columns:
            raise ValueError("DataFrame must contain a 'signal' column.")

        mfe_col = f"forward_mfe_{self.forward_window}"
        mae_col = f"forward_mae_{self.forward_window}"

        signals_df = df.filter(pl.col("signal")).drop_nulls(subset=[mfe_col, mae_col])

        if signals_df.is_empty():
            logger.warning("No valid signals to summarise.")
            return pl.DataFrame()

        total = signals_df.height
        wins = signals_df.filter(pl.col("win")).height
        confidence = wins / total if total > 0 else 0.0

        summary = pl.DataFrame({
            "total_signals": [total],
            "wins": [wins],
            "losses": [total - wins],
            "confidence_score": [round(confidence, 4)],
            "avg_mfe": [round(float(signals_df[mfe_col].mean()), 4)],  # type: ignore[arg-type]
            "avg_mae": [round(float(signals_df[mae_col].mean()), 4)],  # type: ignore[arg-type]
            "median_mfe": [round(float(signals_df[mfe_col].median()), 4)],  # type: ignore[arg-type]
            "median_mae": [round(float(signals_df[mae_col].median()), 4)],  # type: ignore[arg-type]
            "max_mfe": [round(float(signals_df[mfe_col].max()), 4)],  # type: ignore[arg-type]
            "max_mae": [round(float(signals_df[mae_col].max()), 4)],  # type: ignore[arg-type]
        })

        logger.info(
            "Summary: {} signals, {} wins, confidence {:.2%}",
            total,
            wins,
            confidence,
        )
        return summary

    def trade_log(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract a detailed trade log for every signal occurrence.
        """
        mfe_col = f"forward_mfe_{self.forward_window}"
        mae_col = f"forward_mae_{self.forward_window}"

        cols_to_keep = [
            "timestamp", "ticker", "close",
            "velocity_1m", "accel_1m",
            "vpoc_4h",
            mfe_col, mae_col, "win",
        ]
        present = [c for c in cols_to_keep if c in df.columns]

        log = (
            df.filter(pl.col("signal"))
            .select(present)
            .drop_nulls(subset=[mfe_col, mae_col])
            .sort("timestamp")
        )
        logger.info("Trade log contains {} entries", len(log))
        return log

    # ── Directional Forward Metrics (for Market Impulse) ─────────────────

    def add_directional_forward_metrics(
        self,
        df: pl.DataFrame,
        snapshot_windows: tuple[int, ...] = (30, 60),
    ) -> pl.DataFrame:
        """
        Compute forward MFE/MAE that respect signal direction with
        end-of-day measurement window.

        For long signals:  MFE = max(future highs) − entry
                           MAE = entry − min(future lows)
        For short signals: MFE = entry − min(future lows)
                           MAE = max(future highs) − entry

        Also provides snapshot windows (e.g. 30-min, 60-min).

        Requires: 'close', 'high', 'low', 'timestamp', 'signal',
                  'signal_direction' columns.
        """
        n = len(df)
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        timestamps = df["timestamp"].to_list()

        # Get signal direction (None for non-signal bars)
        direction = df["signal_direction"].to_list()

        # Pre-compute date for each bar to find end-of-day boundaries
        dates = df.select(
            pl.col("timestamp").dt.date().alias("trade_date")
        )["trade_date"].to_list()

        # Build day-end index lookup: for each bar, the last bar of that day
        day_end_idx = {}
        for i in range(n - 1, -1, -1):
            d = dates[i]
            if d not in day_end_idx:
                day_end_idx[d] = i

        # ── End-of-day metrics ──────────────────────────────────────────
        mfe_eod = np.full(n, np.nan)
        mae_eod = np.full(n, np.nan)

        for i in range(n):
            if not direction[i]:
                continue

            d = dates[i]
            end_i = day_end_idx.get(d, i)
            if end_i <= i:
                continue

            future_high = high[i + 1 : end_i + 1]
            future_low = low[i + 1 : end_i + 1]
            entry_price = close[i]

            if len(future_high) == 0:
                continue

            if direction[i] == "long":
                mfe_eod[i] = float(np.max(future_high) - entry_price)
                mae_eod[i] = float(entry_price - np.min(future_low))
            elif direction[i] == "short":
                mfe_eod[i] = float(entry_price - np.min(future_low))
                mae_eod[i] = float(np.max(future_high) - entry_price)

        df = df.with_columns([
            pl.Series("forward_mfe_eod", mfe_eod),
            pl.Series("forward_mae_eod", mae_eod),
        ])

        # ── Snapshot windows (e.g., 30-min, 60-min) ─────────────────────
        for window in snapshot_windows:
            mfe_w = np.full(n, np.nan)
            mae_w = np.full(n, np.nan)

            for i in range(n):
                if not direction[i]:
                    continue

                end_i = min(i + window, n - 1)
                if end_i <= i:
                    continue

                future_high = high[i + 1 : end_i + 1]
                future_low = low[i + 1 : end_i + 1]
                entry_price = close[i]

                if len(future_high) == 0:
                    continue

                if direction[i] == "long":
                    mfe_w[i] = float(np.max(future_high) - entry_price)
                    mae_w[i] = float(entry_price - np.min(future_low))
                elif direction[i] == "short":
                    mfe_w[i] = float(entry_price - np.min(future_low))
                    mae_w[i] = float(np.max(future_high) - entry_price)

            df = df.with_columns([
                pl.Series(f"forward_mfe_{window}", mfe_w),
                pl.Series(f"forward_mae_{window}", mae_w),
            ])

        # ── Win flag: MFE ≥ 2× MAE (end-of-day) ────────────────────────
        df = df.with_columns(
            (pl.col("forward_mfe_eod") >= 2.0 * pl.col("forward_mae_eod"))
            .alias("win_2to1")
        )

        logger.info(
            "Directional forward metrics added (EOD + snapshots: {})",
            snapshot_windows,
        )
        return df

    def summarise_directional_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Produce a summary report for directional signals,
        broken out by long / short / combined.
        """
        signals_df = (
            df.filter(pl.col("signal"))
            .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod"])
        )

        if signals_df.is_empty():
            logger.warning("No valid directional signals to summarise.")
            return pl.DataFrame()

        rows = []
        for direction_filter, label in [
            (None, "Combined"),
            ("long", "Long"),
            ("short", "Short"),
        ]:
            if direction_filter:
                subset = signals_df.filter(
                    pl.col("signal_direction") == direction_filter
                )
            else:
                subset = signals_df

            if subset.is_empty():
                continue

            total = subset.height
            wins = subset.filter(pl.col("win_2to1")).height
            confidence = wins / total if total > 0 else 0.0

            row = {
                "direction": label,
                "total_signals": total,
                "wins": wins,
                "losses": total - wins,
                "confidence_2to1": round(confidence, 4),
                "avg_mfe_eod": round(float(subset["forward_mfe_eod"].mean()), 4),
                "avg_mae_eod": round(float(subset["forward_mae_eod"].mean()), 4),
                "median_mfe_eod": round(float(subset["forward_mfe_eod"].median()), 4),
                "median_mae_eod": round(float(subset["forward_mae_eod"].median()), 4),
                "avg_mfe_mae_ratio": round(
                    float(subset["forward_mfe_eod"].mean())
                    / max(float(subset["forward_mae_eod"].mean()), 0.0001),
                    2,
                ),
            }

            # Add snapshot window metrics if available
            for w in (30, 60):
                mfe_col = f"forward_mfe_{w}"
                mae_col = f"forward_mae_{w}"
                if mfe_col in subset.columns:
                    valid = subset.drop_nulls(subset=[mfe_col, mae_col])
                    if not valid.is_empty():
                        row[f"avg_mfe_{w}m"] = round(float(valid[mfe_col].mean()), 4)
                        row[f"avg_mae_{w}m"] = round(float(valid[mae_col].mean()), 4)

            rows.append(row)

        summary = pl.DataFrame(rows)
        for _, row_data in enumerate(rows):
            logger.info(
                "Summary [{}]: {} signals, {} wins, confidence {:.2%}, "
                "MFE/MAE ratio {:.2f}x",
                row_data["direction"],
                row_data["total_signals"],
                row_data["wins"],
                row_data["confidence_2to1"],
                row_data["avg_mfe_mae_ratio"],
            )

        return summary

    def directional_trade_log(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract a detailed trade log for directional signal entries.
        """
        cols_to_keep = [
            "timestamp", "ticker", "close", "signal_direction",
            "impulse_regime_5m", "impulse_stage",
            "vma_10",
            "forward_mfe_eod", "forward_mae_eod",
            "forward_mfe_30", "forward_mae_30",
            "forward_mfe_60", "forward_mae_60",
            "win_2to1",
        ]
        present = [c for c in cols_to_keep if c in df.columns]

        log = (
            df.filter(pl.col("signal"))
            .select(present)
            .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod"])
            .sort("timestamp")
        )
        logger.info("Directional trade log contains {} entries", len(log))
        return log

