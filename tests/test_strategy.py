"""Tests for the EMA Momentum Strategy."""

import numpy as np
import polars as pl
import pytest

from src.strategy.ema_momentum import EMAMomentumStrategy


@pytest.fixture
def enriched_df() -> pl.DataFrame:
    """Physics-enriched DataFrame suitable for strategy testing."""
    n = 100
    np.random.seed(7)
    close = np.cumsum(np.random.randn(n) * 0.05) + 100

    return pl.DataFrame({
        "close": close,
        "ema_4": close + 0.1,   # fastest above
        "ema_8": close + 0.05,  # mid
        "ema_12": close,        # slowest
        "vpoc_4h": close - 1.0,  # price above VPOC → location gate True
        "volume": np.random.randint(5000, 50000, n),
        "volume_ma_20": np.full(n, 20000.0),
    })


class TestEMAMomentumStrategy:
    def test_signal_column_added(self, enriched_df: pl.DataFrame) -> None:
        strat = EMAMomentumStrategy()
        result = strat.generate_signals(enriched_df)
        assert "signal" in result.columns

    def test_all_gates_true_produces_signal(self) -> None:
        """When all three gates are satisfied, signal should be True."""
        df = pl.DataFrame({
            "close": [105.0],
            "ema_4": [106.0],
            "ema_8": [105.5],
            "ema_12": [105.0],
            "vpoc_4h": [100.0],
            "volume": [30000],
            "volume_ma_20": [20000.0],
        })
        strat = EMAMomentumStrategy()
        result = strat.generate_signals(df)
        assert result["signal"][0] is True

    def test_broken_ema_stack_no_signal(self) -> None:
        """When EMA stack is inverted, no signal should fire."""
        df = pl.DataFrame({
            "close": [105.0],
            "ema_4": [104.0],   # slower than ema_8: inverted
            "ema_8": [105.5],
            "ema_12": [106.0],
            "vpoc_4h": [100.0],
            "volume": [30000],
            "volume_ma_20": [20000.0],
        })
        strat = EMAMomentumStrategy()
        result = strat.generate_signals(df)
        assert result["signal"][0] is False

    def test_low_volume_no_signal(self) -> None:
        """Volume below MA should block the signal."""
        df = pl.DataFrame({
            "close": [105.0],
            "ema_4": [106.0],
            "ema_8": [105.5],
            "ema_12": [105.0],
            "vpoc_4h": [100.0],
            "volume": [10000],
            "volume_ma_20": [20000.0],  # volume < MA
        })
        strat = EMAMomentumStrategy()
        result = strat.generate_signals(df)
        assert result["signal"][0] is False

    def test_price_below_vpoc_no_signal(self) -> None:
        """Price below VPOC (in gravity well) should block signal."""
        df = pl.DataFrame({
            "close": [99.0],
            "ema_4": [100.0],
            "ema_8": [99.5],
            "ema_12": [99.0],
            "vpoc_4h": [100.0],  # price < vpoc
            "volume": [30000],
            "volume_ma_20": [20000.0],
        })
        strat = EMAMomentumStrategy()
        result = strat.generate_signals(df)
        assert result["signal"][0] is False

    def test_raises_on_missing_columns(self) -> None:
        df = pl.DataFrame({"close": [100.0]})
        strat = EMAMomentumStrategy()
        with pytest.raises(ValueError, match="requires columns"):
            strat.generate_signals(df)
