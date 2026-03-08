"""Tests for the Oracle Metrics Calculator."""

import numpy as np
import polars as pl
import pytest

from src.oracle.metrics import MetricsCalculator


@pytest.fixture
def signal_df() -> pl.DataFrame:
    """DataFrame with signals and price data for forward metrics."""
    n = 50
    np.random.seed(99)
    close = np.cumsum(np.random.randn(n) * 0.1) + 100
    high = close + 0.05
    low = close - 0.05

    signal = [False] * n
    signal[5] = True
    signal[10] = True
    signal[20] = True

    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            pl.datetime(2024, 1, 2, 9, 30),
            pl.datetime(2024, 1, 2, 9, 30) + pl.duration(minutes=n - 1),
            interval="1m",
            eager=True,
        ),
        "ticker": ["SPY"] * n,
        "close": close,
        "high": high,
        "low": low,
        "volume": np.random.randint(1000, 10000, n),
        "velocity_1m": np.diff(close, prepend=close[0]),
        "accel_1m": np.zeros(n),
        "vpoc_4h": close - 0.5,
        "signal": signal,
    })


class TestMetricsCalculator:
    def test_forward_metrics_columns_added(self, signal_df: pl.DataFrame) -> None:
        mc = MetricsCalculator(forward_window=15)
        result = mc.add_forward_metrics(signal_df)
        assert "forward_mfe_15" in result.columns
        assert "forward_mae_15" in result.columns
        assert "win" in result.columns

    def test_mfe_is_positive_for_upward_move(self) -> None:
        # Construct a scenario where price rises after signal
        close = [100.0] + [100.0 + i * 0.1 for i in range(1, 20)]
        high = [c + 0.02 for c in close]
        low = [c - 0.02 for c in close]

        df = pl.DataFrame({
            "close": close,
            "high": high,
            "low": low,
            "signal": [True] + [False] * 19,
        })

        mc = MetricsCalculator(forward_window=15)
        result = mc.add_forward_metrics(df)

        mfe_val = result["forward_mfe_15"][0]
        assert mfe_val is not None and mfe_val > 0

    def test_summary_confidence_range(self, signal_df: pl.DataFrame) -> None:
        mc = MetricsCalculator(forward_window=15)
        df = mc.add_forward_metrics(signal_df)
        summary = mc.summarise_signals(df)

        if not summary.is_empty():
            conf = summary["confidence_score"][0]
            assert 0.0 <= conf <= 1.0

    def test_trade_log_filters_signals_only(self, signal_df: pl.DataFrame) -> None:
        mc = MetricsCalculator(forward_window=15)
        df = mc.add_forward_metrics(signal_df)
        log = mc.trade_log(df)

        # Should have at most as many entries as signals
        signal_count = signal_df.filter(pl.col("signal")).height
        assert len(log) <= signal_count
