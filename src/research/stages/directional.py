"""Shared directional metric helpers for reusable research stages."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from src.config import settings
from src.oracle.metrics import MetricsCalculator

DEFAULT_DIRECTIONAL_SNAPSHOT_WINDOWS: tuple[int, ...] = (30, 60)


def directional_metric_column_names(window: int | None) -> tuple[str, str]:
    if window is None:
        return "forward_mfe_eod", "forward_mae_eod"
    return f"forward_mfe_{window}", f"forward_mae_{window}"


def available_directional_snapshot_windows(df: pl.DataFrame) -> tuple[int, ...]:
    mfe_windows = {
        int(column.removeprefix("forward_mfe_"))
        for column in df.columns
        if column.startswith("forward_mfe_") and column.removeprefix("forward_mfe_").isdigit()
    }
    mae_windows = {
        int(column.removeprefix("forward_mae_"))
        for column in df.columns
        if column.startswith("forward_mae_") and column.removeprefix("forward_mae_").isdigit()
    }
    return tuple(sorted(mfe_windows & mae_windows))


def resolve_evaluation_window(
    metrics: MetricsCalculator | None = None,
    evaluation_window: int | None = None,
) -> int | None:
    if evaluation_window is None:
        return None
    resolved = evaluation_window
    if resolved <= 0:
        raise ValueError("evaluation_window must be a positive integer.")
    return resolved


def canonical_directional_snapshot_windows(
    *,
    metrics: MetricsCalculator,
    evaluation_window: int | None = None,
    reporting_windows: Iterable[int] = DEFAULT_DIRECTIONAL_SNAPSHOT_WINDOWS,
) -> tuple[int, ...]:
    resolved_window = resolve_evaluation_window(metrics, evaluation_window)
    windows = set(reporting_windows)
    if resolved_window is not None:
        windows.add(resolved_window)
    return tuple(sorted(window for window in windows if window > 0))


def resolve_directional_metric_columns(
    df: pl.DataFrame,
    *,
    evaluation_window: int | None = None,
    metrics: MetricsCalculator | None = None,
    allow_eod_fallback: bool = True,
) -> tuple[str, str, int | None]:
    resolved_window = resolve_evaluation_window(metrics, evaluation_window)
    if resolved_window is None:
        if "forward_mfe_eod" in df.columns and "forward_mae_eod" in df.columns:
            return "forward_mfe_eod", "forward_mae_eod", None
        available = available_directional_snapshot_windows(df)
        if available:
            fallback_window = min(available)
            fallback_mfe, fallback_mae = directional_metric_column_names(fallback_window)
            return fallback_mfe, fallback_mae, fallback_window
        raise ValueError("Directional evaluation columns are missing for end-of-day evaluation.")

    mfe_col, mae_col = directional_metric_column_names(resolved_window)
    if mfe_col in df.columns and mae_col in df.columns:
        return mfe_col, mae_col, resolved_window

    if allow_eod_fallback and "forward_mfe_eod" in df.columns and "forward_mae_eod" in df.columns:
        return "forward_mfe_eod", "forward_mae_eod", None

    raise ValueError(
        "Directional evaluation columns are missing for the requested horizon "
        f"({resolved_window} bars)."
    )
