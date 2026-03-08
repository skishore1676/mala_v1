"""Timezone utilities for converting stored UTC timestamps to ET session time."""

from __future__ import annotations

import polars as pl


def et_timestamp_expr(col: str = "timestamp") -> pl.Expr:
    """
    Return timestamp converted to America/New_York.

    Stored bars are treated as UTC; this helper safely handles
    both naive and UTC-aware timestamp columns.
    """
    return (
        pl.col(col)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/New_York")
    )


def et_time_expr(col: str = "timestamp") -> pl.Expr:
    """Return ET time-of-day expression for a timestamp column."""
    return et_timestamp_expr(col).dt.time()


def et_date_expr(col: str = "timestamp") -> pl.Expr:
    """Return ET calendar-date expression for a timestamp column."""
    return et_timestamp_expr(col).dt.date()
