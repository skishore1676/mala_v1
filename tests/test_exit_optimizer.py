from __future__ import annotations

from datetime import datetime, timezone, date

import polars as pl

from src.research.exit_optimizer import optimize_underlying_exit
from src.strategy.market_impulse import MarketImpulseStrategy


def test_optimize_underlying_exit_selects_supported_policy() -> None:
    strategy = MarketImpulseStrategy(entry_buffer_minutes=3, entry_window_minutes=60, regime_timeframe="1h")
    timestamps = [
        datetime(2026, 3, 16, 13, 35, tzinfo=timezone.utc),
        datetime(2026, 3, 16, 13, 36, tzinfo=timezone.utc),
        datetime(2026, 3, 16, 13, 37, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 13, 35, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 13, 36, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 13, 37, tzinfo=timezone.utc),
    ]
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * len(timestamps),
            "open": [100.0, 100.2, 100.4, 101.0, 101.1, 101.2],
            "high": [100.3, 100.6, 100.8, 101.3, 101.6, 101.8],
            "low": [99.9, 99.95, 100.2, 100.9, 100.95, 101.1],
            "close": [100.2, 100.5, 100.7, 101.1, 101.4, 101.6],
            "volume": [1000.0] * len(timestamps),
            "vma_10": [100.0, 100.1, 100.3, 100.9, 101.0, 101.2],
            "impulse_regime_1h": ["bullish"] * len(timestamps),
        }
    )

    result = optimize_underlying_exit(
        strategy_key="market_impulse",
        symbol="SPY",
        direction="long",
        strategy=strategy,
        enriched_frame=frame,
        holdout_start=date(2026, 3, 16),
        holdout_end=date(2026, 3, 17),
        catastrophe_exit_params={"stop_loss_pct": 0.45, "hard_flat_time_et": "15:55"},
    )

    assert result is not None
    assert result.thesis_exit_anchor == "underlying"
    assert result.catastrophe_exit_anchor == "option_premium"
    assert result.selected_policy_name
    assert result.selected_metrics["trade_count"] >= 1
