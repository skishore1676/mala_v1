"""Tests for strategy factory helper."""

from datetime import time

from src.strategy.base import required_feature_union
from src.strategy.factory import available_strategy_names, build_strategy, build_strategy_by_name
from src.strategy.jerk_pivot_momentum import JerkPivotMomentumStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.market_impulse import MarketImpulseStrategy
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy


def test_build_opening_drive_v2_by_name() -> None:
    strat = build_strategy_by_name("Opening Drive v2 (Short Continue)")
    assert strat.name == "Opening Drive v2 (Short Continue)"


def test_build_jerk_pivot_tight_by_name() -> None:
    strat = build_strategy_by_name("Jerk-Pivot Momentum (tight)")
    assert strat.name == "Jerk-Pivot Momentum (tight)"


def test_build_jerk_pivot_tight_ignores_reward_risk_ratio_override() -> None:
    strat = build_strategy(
        "Jerk-Pivot Momentum (tight)",
        {"reward_risk_ratio": 1.5, "vpoc_proximity_pct": 0.002},
    )
    assert strat.name == "Jerk-Pivot Momentum (tight)"
    assert strat.vpoc_proximity_pct == 0.002


def test_build_strategy_with_override_params() -> None:
    strat = build_strategy(
        "Elastic Band Reversion",
        {"z_score_threshold": 1.25, "z_score_window": 360, "use_directional_mass": True},
    )
    assert strat.name == "Elastic Band z=1.25/w=360+dm"


def test_available_strategy_names_includes_research_candidates() -> None:
    names = available_strategy_names()
    assert "Jerk-Pivot Momentum (tight)" in names
    assert "Opening Drive v2 (Short Continue)" in names
    assert "Market Impulse (Cross & Reclaim)" in names


def test_build_market_impulse_with_timeframe_override() -> None:
    strategy = build_strategy(
        "Market Impulse (Cross & Reclaim)",
        {"regime_timeframe": "15m", "entry_window_minutes": 90},
    )

    assert isinstance(strategy, MarketImpulseStrategy)
    assert strategy.regime_timeframe == "15m"
    assert strategy.evaluation_mode == "directional"


def test_required_feature_union_combines_strategy_dependencies() -> None:
    strategies = [
        build_strategy("Elastic Band Reversion"),
        build_strategy("Opening Drive Classifier"),
    ]

    features = required_feature_union(strategies)

    assert "vpoc_4h" in features
    assert "velocity_1m" in features
    assert "timestamp" in features


def test_jerk_pivot_strategy_accepts_serialized_time_config() -> None:
    strategy = JerkPivotMomentumStrategy(session_start="09:35", session_end="15:30")
    assert strategy.session_start == time(9, 35)
    assert strategy.session_end == time(15, 30)


def test_kinematic_ladder_strategy_accepts_serialized_time_config() -> None:
    strategy = KinematicLadderStrategy(session_start="09:35", session_end="15:30")
    assert strategy.session_start == time(9, 35)
    assert strategy.session_end == time(15, 30)


def test_opening_drive_strategy_accepts_serialized_time_config() -> None:
    strategy = OpeningDriveClassifierStrategy(market_open="09:30")
    assert strategy.market_open == time(9, 30)
