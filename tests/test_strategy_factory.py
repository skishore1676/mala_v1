"""Tests for strategy factory helper."""

from src.strategy.factory import build_strategy_by_name


def test_build_opening_drive_v2_by_name() -> None:
    strat = build_strategy_by_name("Opening Drive v2 (Short Continue)")
    assert strat.name == "Opening Drive v2 (Short Continue)"

