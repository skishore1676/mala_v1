"""Strategy factory helpers used by evaluation and orchestration layers."""

from __future__ import annotations

import re
from typing import Any, Callable

from src.config import settings
from src.strategy.base import BaseStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.jerk_pivot_momentum import JerkPivotMomentumStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.market_impulse import MarketImpulseStrategy
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy
from src.strategy.regime_router import RegimeRouterStrategy


StrategyBuilder = Callable[[dict[str, Any] | None], BaseStrategy]


def _build_elastic(params: dict[str, Any] | None = None) -> BaseStrategy:
    return ElasticBandReversionStrategy(**(params or {}))


def _build_kinematic(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "regime_window": 30,
        "accel_window": 10,
        "volume_multiplier": 1.05,
        "volume_ma_period": settings.volume_ma_period,
        "use_time_filter": True,
    }
    return KinematicLadderStrategy(**(defaults | (params or {})))


def _build_compression(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "compression_window": 20,
        "breakout_lookback": 20,
        "compression_factor": 0.85,
        "volume_ma_period": settings.volume_ma_period,
        "volume_multiplier": 1.15,
        "use_time_filter": True,
    }
    return CompressionBreakoutStrategy(**(defaults | (params or {})))


def _build_router(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "vol_short_window": 20,
        "vol_long_window": 60,
        "trend_vel_window": 30,
        "trend_vol_ratio": 1.0,
        "compression_vol_ratio": 0.9,
        "trend_velocity_floor": 0.015,
    }
    return RegimeRouterStrategy(**(defaults | (params or {})))


def _build_opening_drive(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "opening_window_minutes": 25,
        "entry_start_offset_minutes": 30,
        "entry_end_offset_minutes": 120,
        "min_drive_return_pct": 0.0015,
        "volume_multiplier": 1.2,
    }
    return OpeningDriveClassifierStrategy(**(defaults | (params or {})))


def _build_opening_drive_v2(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "opening_window_minutes": 25,
        "entry_start_offset_minutes": 30,
        "entry_end_offset_minutes": 120,
        "min_drive_return_pct": 0.0020,
        "breakout_buffer_pct": 0.0005,
        "volume_multiplier": 1.4,
        "allow_long": False,
        "allow_short": True,
        "enable_continue": True,
        "enable_fail": False,
        "strategy_label": "Opening Drive v2 (Short Continue)",
    }
    return OpeningDriveClassifierStrategy(**(defaults | (params or {})))


def _build_jerk_pivot_tight(params: dict[str, Any] | None = None) -> BaseStrategy:
    cleaned = dict(params or {})
    # `reward_risk_ratio` is tracked in repo memory for evaluation, but it is not
    # a constructor argument for the strategy itself.
    cleaned.pop("reward_risk_ratio", None)
    defaults = {
        "vpoc_proximity_pct": 0.002,
        "jerk_lookback": 10,
        "volume_multiplier": 1.3,
        "use_volume_filter": True,
        "strategy_label": "Jerk-Pivot Momentum (tight)",
    }
    return JerkPivotMomentumStrategy(**(defaults | cleaned))


def _build_market_impulse(params: dict[str, Any] | None = None) -> BaseStrategy:
    defaults = {
        "entry_buffer_minutes": 3,
        "entry_window_minutes": 60,
        "regime_timeframe": "5m",
    }
    return MarketImpulseStrategy(**(defaults | (params or {})))


_NAMED_BUILDERS: dict[str, StrategyBuilder] = {
    "Elastic Band Reversion": _build_elastic,
    "Kinematic Ladder": _build_kinematic,
    "Compression Expansion Breakout": _build_compression,
    "Regime Router (Kinematic + Compression)": _build_router,
    "Opening Drive Classifier": _build_opening_drive,
    "Opening Drive v2 (Short Continue)": _build_opening_drive_v2,
    "Jerk-Pivot Momentum (tight)": _build_jerk_pivot_tight,
    "Market Impulse (Cross & Reclaim)": _build_market_impulse,
}


def available_strategy_names() -> tuple[str, ...]:
    return tuple(sorted(_NAMED_BUILDERS))


def build_strategy(strategy_name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    builder = _NAMED_BUILDERS.get(strategy_name)
    if builder is None:
        raise ValueError(f"Unsupported strategy name: {strategy_name}")
    return builder(params)


def build_strategy_by_name(strategy_name: str) -> BaseStrategy:
    """Backward-compatible parser for scripts and stored strategy names."""
    # Parse parametric Elastic Band: "Elastic Band z=1.0/w=240+dm"
    if strategy_name.startswith("Elastic Band z="):
        match = re.search(r"z=([\d\.]+)/w=(\d+)(\+dm)?", strategy_name)
        if match:
            return ElasticBandReversionStrategy(
                z_score_threshold=float(match.group(1)),
                z_score_window=int(match.group(2)),
                use_directional_mass=bool(match.group(3)),
            )

    # Parse parametric Kinematic Ladder: "Kinematic Ladder rw=20/aw=8-vol"
    if strategy_name.startswith("Kinematic Ladder rw="):
        match = re.search(r"rw=(\d+)/aw=(\d+)([+-]vol)?", strategy_name)
        if match:
            vol_flag = match.group(3)
            return KinematicLadderStrategy(
                regime_window=int(match.group(1)),
                accel_window=int(match.group(2)),
                use_volume_filter=(vol_flag == "+vol" if vol_flag else True),
                volume_multiplier=1.05,
                volume_ma_period=settings.volume_ma_period,
                use_time_filter=True,
            )

    return build_strategy(strategy_name)
