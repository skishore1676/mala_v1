"""Strategy factory helpers used by evaluation scripts."""

from __future__ import annotations

from src.config import settings
from src.strategy.base import BaseStrategy
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.regime_router import RegimeRouterStrategy
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy


def build_strategy_by_name(strategy_name: str) -> BaseStrategy:
    if strategy_name == "Elastic Band Reversion":
        return ElasticBandReversionStrategy(
            z_score_threshold=2.0,
            z_score_window=240,
        )
    if strategy_name == "Kinematic Ladder":
        return KinematicLadderStrategy(
            regime_window=30,
            accel_window=10,
            volume_multiplier=1.05,
            volume_ma_period=settings.volume_ma_period,
            use_time_filter=True,
        )
    if strategy_name == "Compression Expansion Breakout":
        return CompressionBreakoutStrategy(
            compression_window=20,
            breakout_lookback=20,
            compression_factor=0.85,
            volume_ma_period=settings.volume_ma_period,
            volume_multiplier=1.15,
            use_time_filter=True,
        )
    if strategy_name == "Regime Router (Kinematic + Compression)":
        return RegimeRouterStrategy(
            vol_short_window=20,
            vol_long_window=60,
            trend_vel_window=30,
            trend_vol_ratio=1.0,
            compression_vol_ratio=0.9,
            trend_velocity_floor=0.015,
        )
    if strategy_name == "Opening Drive Classifier":
        return OpeningDriveClassifierStrategy(
            opening_window_minutes=25,
            entry_start_offset_minutes=25,
            entry_end_offset_minutes=120,
            min_drive_return_pct=0.0015,
            volume_multiplier=1.2,
        )
    if strategy_name == "Opening Drive v2 (Short Continue)":
        return OpeningDriveClassifierStrategy(
            opening_window_minutes=25,
            entry_start_offset_minutes=25,
            entry_end_offset_minutes=120,
            min_drive_return_pct=0.0020,
            breakout_buffer_pct=0.0005,
            volume_multiplier=1.4,
            allow_long=False,
            allow_short=True,
            enable_continue=True,
            enable_fail=False,
            strategy_label="Opening Drive v2 (Short Continue)",
        )
    raise ValueError(f"Unsupported strategy name: {strategy_name}")

