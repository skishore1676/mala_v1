"""
Newton Physics Engine

Pre-calculates kinematic derivatives of price action:
  - Velocity   (1st derivative  dP/dt)
  - Acceleration (2nd derivative  dv/dt)
  - Jerk       (3rd derivative  da/dt)
  - Gravity    (VPOC – Volume Point of Control)
  - EMA stack  (configurable periods)
  - Market Impulse (VMA + VWMA regime, multi-timeframe)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import polars as pl
from loguru import logger

from src.config import settings


class PhysicsEngine:
    """
    Transforms a raw OHLCV DataFrame into a "Physics-Enriched" DataFrame
    by appending kinematic columns.
    """

    def __init__(
        self,
        vpoc_lookback: int = settings.vpoc_lookback_bars,
        ema_periods: Optional[List[int]] = None,
        volume_ma_period: int = settings.volume_ma_period,
    ) -> None:
        self.vpoc_lookback = vpoc_lookback
        self.ema_periods = ema_periods or list(settings.ema_periods)
        self.volume_ma_period = volume_ma_period

    # ── Public ───────────────────────────────────────────────────────────

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Accept a raw OHLCV DataFrame (must have 'close', 'high', 'low',
        'volume' columns) and return a new DataFrame with physics columns
        appended.
        """
        required = {"close", "high", "low", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        logger.info("Enriching {} bars with physics columns …", len(df))

        df = self._add_velocity(df)
        df = self._add_acceleration(df)
        df = self._add_jerk(df)
        df = self._add_emas(df)
        df = self._add_volume_ma(df)
        df = self._add_vpoc(df)

        logger.info("Physics enrichment complete – {} columns total", len(df.columns))
        return df

    # ── Velocity ─────────────────────────────────────────────────────────

    @staticmethod
    def _add_velocity(df: pl.DataFrame) -> pl.DataFrame:
        """Velocity = close_t − close_{t-1} (1-period ROC in absolute terms)."""
        return df.with_columns(
            (pl.col("close") - pl.col("close").shift(1)).alias("velocity_1m")
        )

    # ── Acceleration ─────────────────────────────────────────────────────

    @staticmethod
    def _add_acceleration(df: pl.DataFrame) -> pl.DataFrame:
        """Acceleration = velocity_t − velocity_{t-1}."""
        return df.with_columns(
            (pl.col("velocity_1m") - pl.col("velocity_1m").shift(1)).alias("accel_1m")
        )

    # ── Jerk ─────────────────────────────────────────────────────────────

    @staticmethod
    def _add_jerk(df: pl.DataFrame) -> pl.DataFrame:
        """Jerk = accel_t − accel_{t-1}."""
        return df.with_columns(
            (pl.col("accel_1m") - pl.col("accel_1m").shift(1)).alias("jerk_1m")
        )

    # ── EMAs ─────────────────────────────────────────────────────────────

    def _add_emas(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Exponential Moving Averages for each configured period."""
        exprs = []
        for period in self.ema_periods:
            exprs.append(
                pl.col("close")
                .ewm_mean(span=period, adjust=False)
                .alias(f"ema_{period}")
            )
        return df.with_columns(exprs)

    # ── Volume Moving Average ────────────────────────────────────────────

    def _add_volume_ma(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rolling mean of volume for the configured window."""
        return df.with_columns(
            pl.col("volume")
            .rolling_mean(window_size=self.volume_ma_period)
            .alias(f"volume_ma_{self.volume_ma_period}")
        )

    # ── VPOC (Volume Point of Control) ───────────────────────────────────

    def _add_vpoc(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rolling VPOC: for each bar, find the price level with the highest
        traded volume in the preceding *vpoc_lookback* bars.

        We discretise prices into bins of 0.01 and tally volume by bin,
        then pick the bin centre with the maximum volume.
        """
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(df)

        vpoc = np.full(n, np.nan)

        for i in range(self.vpoc_lookback, n):
            window_start = i - self.vpoc_lookback
            # Use typical price = (H + L + C) / 3 as the representative price
            typical = (high[window_start:i] + low[window_start:i] + close[window_start:i]) / 3.0
            vol_slice = volume[window_start:i]

            # Bin prices to 2 decimal places
            price_bins = np.round(typical, 2)
            # Accumulate volume per price bin
            unique_prices, inverse = np.unique(price_bins, return_inverse=True)
            vol_by_price = np.zeros(len(unique_prices))
            np.add.at(vol_by_price, inverse, vol_slice)

            vpoc[i] = unique_prices[np.argmax(vol_by_price)]

        return df.with_columns(pl.Series("vpoc_4h", vpoc))

    # ── Market Impulse (Multi-Timeframe) ─────────────────────────────────

    def enrich_market_impulse(
        self,
        df: pl.DataFrame,
        vma_length: int = 10,
        vwma_periods: tuple[int, ...] = (8, 21, 34),
        market_open: tuple[int, int] = (9, 30),
        market_close: tuple[int, int] = (16, 0),
    ) -> pl.DataFrame:
        """
        Add Market Impulse columns using a multi-timeframe approach:

          0. Filter to market hours only (9:30–16:00 ET) to avoid
             pre/post-market noise corrupting the adaptive VMA.
          1. Compute VMA, VWMA, regime, stage on the 1-min bars.
          2. Resample to 5-min bars, compute regime on 5-min data.
          3. Join the 5-min regime AND vma_10_5m back to 1-min bars.

        Requires 'timestamp', 'close', 'open', 'high', 'low', 'volume'.
        """
        from datetime import time as dt_time
        from src.newton.market_impulse import enrich_impulse_columns

        # ── Filter to market hours ──────────────────────────────────────
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column — skipping market-hours filter.")
        else:
            mkt_open = dt_time(market_open[0], market_open[1])
            mkt_close = dt_time(market_close[0], market_close[1])

            before = len(df)
            df = df.filter(
                (pl.col("timestamp").dt.time() >= mkt_open)
                & (pl.col("timestamp").dt.time() <= mkt_close)
            )
            dropped = before - len(df)
            if dropped > 0:
                logger.info(
                    "Filtered to market hours ({} – {}): dropped {} bars, "
                    "{} bars remaining",
                    mkt_open, mkt_close, dropped, len(df),
                )

        # ── 1-min impulse columns ───────────────────────────────────────
        df = enrich_impulse_columns(
            df,
            vma_length=vma_length,
            vwma_periods=vwma_periods,
            suffix="",
        )

        # ── Resample to 5-min bars ──────────────────────────────────────
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column — skipping 5-min resample.")
            return df

        df_5m = (
            df.sort("timestamp")
            .group_by_dynamic("timestamp", every="5m")
            .agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ])
        )

        # ── 5-min impulse columns ───────────────────────────────────────
        df_5m = enrich_impulse_columns(
            df_5m,
            vma_length=vma_length,
            vwma_periods=vwma_periods,
            suffix="_5m",
        )

        # Keep regime, stage, AND the 5-min VMA value for exit logic
        vma_5m_col = f"vma_{vma_length}_5m"
        join_cols = [
            "timestamp",
            "impulse_regime_5m",
            "impulse_stage_5m",
            vma_5m_col,
        ]
        # Only keep columns that exist
        join_cols = [c for c in join_cols if c in df_5m.columns]
        df_5m_join = df_5m.select(join_cols).rename({"timestamp": "ts_5m"})

        # ── Join 5-min data back to 1-min bars ──────────────────────────
        df = df.sort("timestamp")
        df_5m_join = df_5m_join.sort("ts_5m")

        df = df.join_asof(
            df_5m_join,
            left_on="timestamp",
            right_on="ts_5m",
            strategy="backward",
        )

        # Clean up the join key
        if "ts_5m" in df.columns:
            df = df.drop("ts_5m")

        logger.info(
            "Multi-timeframe Market Impulse enrichment complete – "
            "{} 1-min bars, {} 5-min bars (market hours only)",
            len(df),
            len(df_5m),
        )
        return df

