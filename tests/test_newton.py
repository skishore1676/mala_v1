"""Tests for the Newton Physics Engine."""

import numpy as np
import polars as pl
import pytest

from src.newton.engine import PhysicsEngine


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
            "volume_ma_20", "vpoc_4h",
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

    def test_raises_on_missing_columns(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0]})
        engine = PhysicsEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.enrich(df)
