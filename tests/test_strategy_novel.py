"""Tests for newly added directional strategies."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.regime_router import RegimeRouterStrategy


def _minute_series(n: int, start: datetime = datetime(2025, 1, 2, 10, 0)) -> list[datetime]:
    return [start + timedelta(minutes=i) for i in range(n)]


class TestElasticBandReversionStrategy:
    def test_generates_long_and_short_signals(self) -> None:
        df = pl.DataFrame({
            "close": [100.0, 100.1, 99.9, 98.0, 102.0],
            "vpoc_4h": [100.0, 100.0, 100.0, 100.0, 100.0],
            "velocity_1m": [0.0, 0.1, -0.1, -0.3, 0.3],
            "jerk_1m": [0.0, -0.1, 0.1, 0.2, -0.2],
            "directional_mass": [0.0, 5.0, 10.0, 500.0, -500.0],
        })

        strat = ElasticBandReversionStrategy(
            z_score_threshold=1.0,
            z_score_window=3,
        )
        out = strat.generate_signals(df)

        directions = [x for x in out["signal_direction"].to_list() if x is not None]
        assert "long" in directions
        assert "short" in directions

    def test_missing_columns_raise(self) -> None:
        strat = ElasticBandReversionStrategy()
        with pytest.raises(ValueError, match="requires columns"):
            strat.generate_signals(pl.DataFrame({"close": [100.0]}))


class TestKinematicLadderStrategy:
    def test_generates_long_signal(self) -> None:
        df = pl.DataFrame({
            "timestamp": _minute_series(5),
            "close": [100.0, 100.2, 100.3, 100.22, 100.25],
            "ema_4": [100.1, 100.25, 100.35, 100.3, 100.32],
            "ema_8": [99.95, 100.1, 100.22, 100.24, 100.27],
            "ema_12": [99.8, 99.95, 100.08, 100.12, 100.16],
            "velocity_1m": [0.1, 0.1, 0.1, -0.05, 0.03],
            "accel_1m": [0.02, 0.02, 0.02, -0.01, 0.01],
            "jerk_1m": [0.01, 0.01, 0.01, -0.02, 0.02],
            "volume": [2000, 2200, 2400, 2600, 2800],
            "volume_ma_20": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        })

        strat = KinematicLadderStrategy(
            regime_window=3,
            accel_window=3,
            volume_ma_period=20,
            volume_multiplier=1.0,
            use_time_filter=False,
        )

        out = strat.generate_signals(df)

        assert "signal" in out.columns
        assert "signal_direction" in out.columns
        assert "_vel_regime" not in out.columns
        assert "_acc_regime" not in out.columns
        assert "long" in [x for x in out["signal_direction"].to_list() if x is not None]

    def test_missing_columns_raise(self) -> None:
        strat = KinematicLadderStrategy(use_time_filter=False)
        with pytest.raises(ValueError, match="requires columns"):
            strat.generate_signals(pl.DataFrame({"close": [100.0]}))


class TestCompressionBreakoutStrategy:
    def test_generates_signal_columns(self) -> None:
        df = pl.DataFrame({
            "timestamp": _minute_series(12),
            "close": [100.0, 100.02, 100.01, 100.03, 100.02, 100.01, 100.02, 100.03, 100.02, 100.01, 100.0, 100.5],
            "high": [100.1, 100.06, 100.05, 100.08, 100.06, 100.05, 100.06, 100.08, 100.05, 100.04, 100.03, 100.6],
            "low": [99.9, 99.98, 99.97, 99.99, 99.98, 99.97, 99.98, 99.99, 99.97, 99.96, 99.95, 100.0],
            "ema_8": [100.0] * 11 + [100.2],
            "ema_12": [99.95] * 12,
            "velocity_1m": [0.01] * 11 + [0.2],
            "volume": [1000] * 11 + [5000],
            "volume_ma_20": [800.0] * 12,
        })
        strat = CompressionBreakoutStrategy(
            compression_window=3,
            breakout_lookback=3,
            compression_factor=1.0,
            volume_ma_period=20,
            volume_multiplier=1.0,
            use_time_filter=False,
        )
        out = strat.generate_signals(df)
        assert "signal" in out.columns
        assert "signal_direction" in out.columns
        assert out["signal"][-1] is True

    def test_missing_columns_raise(self) -> None:
        strat = CompressionBreakoutStrategy(use_time_filter=False)
        with pytest.raises(ValueError, match="requires columns"):
            strat.generate_signals(pl.DataFrame({"close": [100.0]}))


class TestRegimeRouterStrategy:
    def test_generates_routed_signal_columns(self) -> None:
        n = 80
        close = [100.0 + 0.02 * i for i in range(n - 1)] + [103.0]
        df = pl.DataFrame({
            "timestamp": _minute_series(n),
            "close": close,
            "high": [c + 0.1 for c in close],
            "low": [c - 0.1 for c in close],
            "ema_4": [c + 0.05 for c in close],
            "ema_8": [c for c in close],
            "ema_12": [c - 0.05 for c in close],
            "velocity_1m": [0.02] * (n - 1) + [0.8],
            "accel_1m": [0.0] * (n - 2) + [0.02, 0.4],
            "jerk_1m": [0.0] * (n - 2) + [0.02, 0.2],
            "volume": [1500] * (n - 1) + [9000],
            "volume_ma_20": [1000.0] * n,
        })

        strat = RegimeRouterStrategy(
            kinematic=KinematicLadderStrategy(use_time_filter=False, volume_multiplier=1.0),
            compression=CompressionBreakoutStrategy(use_time_filter=False, volume_multiplier=1.0),
            vol_short_window=8,
            vol_long_window=20,
            trend_vel_window=10,
            trend_vol_ratio=0.9,
            compression_vol_ratio=0.95,
            trend_velocity_floor=0.005,
        )

        out = strat.generate_signals(df)
        assert "signal" in out.columns
        assert "signal_direction" in out.columns
        assert "route_regime" in out.columns

    def test_missing_columns_raise(self) -> None:
        strat = RegimeRouterStrategy()
        with pytest.raises(ValueError, match="requires columns"):
            strat.generate_signals(pl.DataFrame({"close": [100.0]}))
