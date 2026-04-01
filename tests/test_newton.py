"""Tests for the Newton Physics Engine."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.newton.engine import PhysicsEngine
from src.newton.resampler import TimeframeResampler
from src.newton.transforms import JerkTransform, MarketImpulseTransform
from src.strategy.base import required_feature_union
from src.strategy.market_impulse import MarketImpulseStrategy


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Create a small synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 300  # enough rows for VPOC lookback (240)
    close = np.cumsum(np.random.randn(n) * 0.1) + 100
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    volume = np.random.randint(1000, 50000, size=n)

    return pl.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "open": close - np.random.randn(n) * 0.02,
        "volume": volume,
    })


class TestPhysicsEngine:
    def test_enrich_adds_all_columns(self, sample_ohlcv: pl.DataFrame) -> None:
        engine = PhysicsEngine(vpoc_lookback=50, ema_periods=[4, 8, 12])
        result = engine.enrich(sample_ohlcv)

        expected_cols = {
            "velocity_1m", "accel_1m", "jerk_1m",
            "ema_4", "ema_8", "ema_12",
            "volume_ma_20", "internal_strength",
            "directional_mass", "directional_mass_ma_20", "vpoc_4h",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_velocity_is_first_diff(self, sample_ohlcv: pl.DataFrame) -> None:
        engine = PhysicsEngine(vpoc_lookback=50)
        result = engine.enrich(sample_ohlcv)

        close = result["close"].to_numpy()
        velocity = result["velocity_1m"].to_numpy()
        # velocity[1] should equal close[1] - close[0]
        np.testing.assert_almost_equal(velocity[1], close[1] - close[0], decimal=10)

    def test_acceleration_is_second_diff(self, sample_ohlcv: pl.DataFrame) -> None:
        engine = PhysicsEngine(vpoc_lookback=50)
        result = engine.enrich(sample_ohlcv)

        vel = result["velocity_1m"].to_numpy()
        accel = result["accel_1m"].to_numpy()
        # accel[2] should equal vel[2] - vel[1]
        np.testing.assert_almost_equal(accel[2], vel[2] - vel[1], decimal=10)

    def test_jerk_is_third_diff(self, sample_ohlcv: pl.DataFrame) -> None:
        engine = PhysicsEngine(vpoc_lookback=50)
        result = engine.enrich(sample_ohlcv)

        accel = result["accel_1m"].to_numpy()
        jerk = result["jerk_1m"].to_numpy()
        np.testing.assert_almost_equal(jerk[3], accel[3] - accel[2], decimal=10)

    def test_ema_columns_count(self, sample_ohlcv: pl.DataFrame) -> None:
        periods = [5, 10, 20, 50]
        engine = PhysicsEngine(vpoc_lookback=50, ema_periods=periods)
        result = engine.enrich(sample_ohlcv)

        for p in periods:
            assert f"ema_{p}" in result.columns

    def test_vpoc_populated_after_lookback(self, sample_ohlcv: pl.DataFrame) -> None:
        lookback = 50
        engine = PhysicsEngine(vpoc_lookback=lookback)
        result = engine.enrich(sample_ohlcv)

        vpoc = result["vpoc_4h"].to_numpy()
        # Before lookback window, VPOC should be NaN
        assert np.isnan(vpoc[lookback - 1])
        # After lookback window, VPOC should be filled
        assert not np.isnan(vpoc[lookback])

    def test_directional_mass_formula_and_zero_range_handling(self) -> None:
        df = pl.DataFrame({
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 10.0, 12.0],
            "low": [9.0, 10.0, 10.0],
            "close": [10.5, 10.0, 11.0],
            "volume": [1000.0, 2000.0, 1500.0],
        })
        engine = PhysicsEngine(vpoc_lookback=2, volume_ma_period=2)
        result = engine.enrich(df)

        strength = result["internal_strength"].to_numpy()
        dmass = result["directional_mass"].to_numpy()
        dmass_ma = result["directional_mass_ma_2"].to_numpy()

        np.testing.assert_almost_equal(strength[0], 0.5, decimal=10)
        np.testing.assert_almost_equal(dmass[0], 500.0, decimal=10)
        np.testing.assert_almost_equal(strength[1], 0.0, decimal=10)  # high == low
        np.testing.assert_almost_equal(dmass[1], 0.0, decimal=10)
        np.testing.assert_almost_equal(dmass_ma[1], 250.0, decimal=10)
        np.testing.assert_almost_equal(dmass_ma[2], 0.0, decimal=10)

    def test_raises_on_missing_columns(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0]})
        engine = PhysicsEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.enrich(df)

    def test_pipeline_resolves_dependencies_for_subset_transforms(
        self,
        sample_ohlcv: pl.DataFrame,
    ) -> None:
        engine = PhysicsEngine(transforms=[JerkTransform()])
        result = engine.enrich(sample_ohlcv)

        assert {"velocity_1m", "accel_1m", "jerk_1m"}.issubset(result.columns)
        assert "vpoc_4h" not in result.columns
        assert "ema_20" not in result.columns

    def test_enrich_for_features_builds_minimal_pipeline(
        self,
        sample_ohlcv: pl.DataFrame,
    ) -> None:
        engine = PhysicsEngine(vpoc_lookback=50, ema_periods=[4, 8, 12])
        result = engine.enrich_for_features(sample_ohlcv, {"jerk_1m", "vpoc_4h"})

        assert {"velocity_1m", "accel_1m", "jerk_1m", "vpoc_4h"}.issubset(result.columns)
        assert "ema_4" not in result.columns
        assert "directional_mass" not in result.columns

    def test_parameterized_velocity_uses_multi_bar_difference(
        self,
        sample_ohlcv: pl.DataFrame,
    ) -> None:
        engine = PhysicsEngine(vpoc_lookback=50)
        result = engine.enrich_for_features(sample_ohlcv, {"velocity_3"})

        close = result["close"].to_numpy()
        velocity = result["velocity_3"].to_numpy()
        np.testing.assert_almost_equal(velocity[3], close[3] - close[0], decimal=10)
        assert "velocity_1m" not in result.columns

    def test_parameterized_jerk_features_resolve_matching_dependencies(
        self,
        sample_ohlcv: pl.DataFrame,
    ) -> None:
        engine = PhysicsEngine(vpoc_lookback=50)
        result = engine.enrich_for_features(sample_ohlcv, {"jerk_3"})

        assert {"velocity_3", "accel_3", "jerk_3"}.issubset(result.columns)
        assert "jerk_1m" not in result.columns

    def test_parameterized_kinematic_transform_specs_are_supported(
        self,
        sample_ohlcv: pl.DataFrame,
    ) -> None:
        engine = PhysicsEngine(transforms=["jerk:3"])
        result = engine.enrich(sample_ohlcv)

        assert {"velocity_3", "accel_3", "jerk_3"}.issubset(result.columns)


def test_resampler_joins_without_lookahead() -> None:
    timestamps = [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(6)]
    base = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5, 6],
            "close": [1, 2, 3, 4, 5, 6],
            "volume": [10, 10, 10, 10, 10, 10],
        }
    )
    resampler = TimeframeResampler()
    five_min = resampler.resample_ohlcv(base, every="5m").with_columns(
        pl.Series("regime_5m", ["first", "second"])
    )

    joined = resampler.join_timeframe_features(
        base,
        five_min,
        every="5m",
        feature_columns=["regime_5m"],
    )
    assert joined["regime_5m"].to_list() == ["first", "first", "first", "first", "first", "second"]


def test_market_impulse_transform_supports_custom_timeframe() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    engine = PhysicsEngine(transforms=[MarketImpulseTransform(timeframe="15m")])
    result = engine.enrich(df)

    assert "impulse_regime_15m" in result.columns
    assert "impulse_stage_15m" in result.columns
    assert "vma_10_15m" in result.columns


def test_market_impulse_transform_rejects_invalid_vwma_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketImpulseTransform(vwma_periods=(34, 21, 8))


def test_market_impulse_transform_rejects_invalid_vwma_count() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        MarketImpulseTransform(vwma_periods=(8, 21))


def test_enrich_for_features_accepts_market_impulse_transform_name() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, {"market_impulse"})

    assert "impulse_regime_5m" in result.columns
    assert "vma_10_5m" in result.columns


def test_enrich_for_features_accepts_parameterized_market_impulse_transform_name() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, {"market_impulse:15m"})

    assert "impulse_regime_15m" in result.columns
    assert "vma_10_15m" in result.columns
    assert "impulse_regime_5m" not in result.columns


def test_enrich_for_features_resolves_market_impulse_from_feature_columns() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, {"impulse_regime_15m", "vma_10"})

    assert "impulse_regime_15m" in result.columns
    assert "vma_10" in result.columns


def test_enrich_for_features_can_build_multiple_market_impulse_timeframes() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(90)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 104, 90),
            "high": np.linspace(100.1, 104.1, 90),
            "low": np.linspace(99.9, 103.9, 90),
            "close": np.linspace(100, 104, 90),
            "volume": np.full(90, 1000.0),
        }
    )
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, {"market_impulse:5m", "market_impulse:15m"})

    assert "impulse_regime_5m" in result.columns
    assert "impulse_regime_15m" in result.columns


def test_market_impulse_strategy_declares_pipeline_resolvable_features() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    strategy = MarketImpulseStrategy()
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, required_feature_union([strategy]))

    assert strategy.required_features.issubset(result.columns)
    assert "vma_10_5m" in result.columns


def test_market_impulse_strategy_can_request_alternate_regime_timeframe() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    strategy = MarketImpulseStrategy(regime_timeframe="15m")
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, required_feature_union([strategy]))

    assert "impulse_regime_15m" in result.columns
    assert strategy.required_features.issubset(result.columns)


def test_market_impulse_strategy_can_request_alternate_vwma_stack() -> None:
    timestamps = [datetime(2025, 1, 2, 14, 30) + timedelta(minutes=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(100, 102, 40),
            "high": np.linspace(100.1, 102.1, 40),
            "low": np.linspace(99.9, 101.9, 40),
            "close": np.linspace(100, 102, 40),
            "volume": np.full(40, 1000.0),
        }
    )
    strategy = MarketImpulseStrategy(vwma_periods=(5, 13, 21))
    engine = PhysicsEngine()
    result = engine.enrich_for_features(df, required_feature_union([strategy]))

    assert "vwma_5" in result.columns
    assert "vwma_13" in result.columns
    assert "vwma_21" in result.columns
    assert "vwma_34" not in result.columns
    assert strategy.required_features.issubset(result.columns)
