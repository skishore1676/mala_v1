"""Tests for pluggable trade-simulator exit policies."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.oracle.trade_simulator import (
    FixedRewardRiskExitPolicy,
    TradeSimulator,
    VmaTrailingExitPolicy,
)


def _base_trade_frame() -> pl.DataFrame:
    timestamps = [datetime(2025, 1, 2, 9, 30) + timedelta(minutes=i) for i in range(5)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.0, 98.0, 98.0, 98.0],
            "high": [100.0, 101.0, 99.0, 98.5, 98.5],
            "low": [100.0, 100.0, 97.0, 97.5, 97.5],
            "signal": [True, False, False, False, False],
            "signal_direction": ["long", None, None, None, None],
        }
    )


def test_trade_simulator_defaults_to_vma_trailing_exit() -> None:
    df = _base_trade_frame().with_columns(pl.Series("vma_10_5m", [99.0, 100.0, 100.0, 100.0, 100.0]))

    result = TradeSimulator().simulate(df)

    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "vma_stop"
    assert trade.exit_price == 98.0
    assert trade.pnl == -2.0


def test_trade_simulator_can_use_fixed_rr_without_vma_columns() -> None:
    timestamps = [datetime(2025, 1, 2, 9, 30) + timedelta(minutes=i) for i in range(4)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.5, 101.0, 100.5],
            "high": [100.0, 102.5, 101.2, 100.8],
            "low": [100.0, 99.8, 100.7, 100.2],
            "signal": [True, False, False, False],
            "signal_direction": ["long", None, None, None],
        }
    )

    result = TradeSimulator(
        exit_policy=FixedRewardRiskExitPolicy(stop_loss=1.0, reward_multiple=2.0)
    ).simulate(df)

    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == 102.0
    assert trade.pnl == 2.0
    assert trade.vma_5m_at_entry == 0.0


def test_fixed_rr_policy_validates_configuration() -> None:
    with pytest.raises(ValueError, match="stop_loss must be positive"):
        FixedRewardRiskExitPolicy(stop_loss=0.0)

    with pytest.raises(ValueError, match="reward_multiple must be positive"):
        FixedRewardRiskExitPolicy(stop_loss=1.0, reward_multiple=0.0)


def test_vma_trailing_policy_requires_vma_column() -> None:
    with pytest.raises(ValueError, match="requires columns"):
        TradeSimulator(exit_policy=VmaTrailingExitPolicy(vma_col="custom_vma")).simulate(
            _base_trade_frame()
        )
