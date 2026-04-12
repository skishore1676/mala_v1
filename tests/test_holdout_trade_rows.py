"""Tests for the per-trade-row extraction helper in holdout.py.

``extract_signal_trade_rows`` is a new pure-polars helper added for
W2.1 (per-regime performance slicing). It shares the same filter
semantics as ``evaluate_df`` — signal==True + non-null mfe/mae +
matching signal_direction — so this test file pins that contract.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from src.research.stages.holdout import extract_signal_trade_rows


def _make_eval_df() -> pl.DataFrame:
    """Small synthetic df_eval mirroring the shape consumed by
    evaluate_df in walk_forward.py — columns: timestamp, close, signal,
    signal_direction, forward_mfe_eod, forward_mae_eod.
    """
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 4, 6, 9, 30),   # Monday, signal=True long
                datetime(2026, 4, 6, 10, 30),  # Monday, no signal
                datetime(2026, 4, 7, 9, 30),   # Tuesday, signal=True short
                datetime(2026, 4, 7, 10, 30),  # Tuesday, signal=True short
                datetime(2026, 4, 8, 9, 30),   # Wednesday, signal=True short but mae=None
            ],
            "close": [100.0, 100.0, 200.0, 200.0, 300.0],
            "signal": [True, False, True, True, True],
            "signal_direction": ["long", None, "short", "short", "short"],
            "forward_mfe_eod": [2.0, None, 3.0, 1.5, 4.0],
            "forward_mae_eod": [1.0, None, 1.0, 1.2, None],
        }
    )


class TestExtractSignalTradeRows:
    def test_short_direction_filters_correctly(self) -> None:
        rows = extract_signal_trade_rows(
            _make_eval_df(),
            direction="short",
            evaluation_window=None,
        )
        # 3 signal rows had signal_direction=short, but one had mae=None
        # so it gets dropped by drop_nulls. Expect 2 rows.
        assert len(rows) == 2
        assert all(r["signal_direction"] == "short" for r in rows)
        assert {r["trade_date"] for r in rows} == {date(2026, 4, 7)}

    def test_long_direction_returns_only_long_signals(self) -> None:
        rows = extract_signal_trade_rows(
            _make_eval_df(),
            direction="long",
            evaluation_window=None,
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["signal_direction"] == "long"
        assert r["trade_date"] == date(2026, 4, 6)
        assert r["mfe"] == 2.0
        assert r["mae"] == 1.0

    def test_combined_direction_includes_both(self) -> None:
        rows = extract_signal_trade_rows(
            _make_eval_df(),
            direction="combined",
            evaluation_window=None,
        )
        # 1 long + 2 shorts (one short dropped for null mae) = 3
        assert len(rows) == 3
        dirs = sorted(r["signal_direction"] for r in rows)
        assert dirs == ["long", "short", "short"]

    def test_empty_when_no_matching_signals(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2026, 4, 6, 9, 30)],
                "close": [100.0],
                "signal": [False],
                "signal_direction": [None],
                "forward_mfe_eod": [None],
                "forward_mae_eod": [None],
            }
        )
        rows = extract_signal_trade_rows(
            df, direction="short", evaluation_window=None
        )
        assert rows == []

    def test_returned_trade_date_is_python_date(self) -> None:
        # catalog_regime_performance.classify_range expects datetime.date
        # keys — not polars.Date or datetime — so confirm the helper
        # normalizes via to_dicts().
        rows = extract_signal_trade_rows(
            _make_eval_df(),
            direction="long",
            evaluation_window=None,
        )
        assert isinstance(rows[0]["trade_date"], date)
