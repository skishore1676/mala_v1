"""
Newton Physics Engine

Compatibility facade over a composable feature pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from loguru import logger

from src.config import settings
from src.newton.transforms import (
    AccelerationTransform,
    DirectionalMassTransform,
    EmaStackTransform,
    FeatureTransform,
    JerkTransform,
    MarketImpulseTransform,
    VelocityTransform,
    VolumeMaTransform,
    VpocTransform,
    transform_names,
)


class PhysicsEngine:
    """Apply Newton feature transforms to raw market data."""

    def __init__(
        self,
        vpoc_lookback: int = settings.vpoc_lookback_bars,
        ema_periods: Sequence[int] | None = None,
        volume_ma_period: int = settings.volume_ma_period,
        transforms: Sequence[FeatureTransform | str] | None = None,
    ) -> None:
        self.vpoc_lookback = vpoc_lookback
        self.ema_periods = tuple(ema_periods or settings.ema_periods)
        self.volume_ma_period = volume_ma_period
        self._registry = self._build_registry()
        selected = transforms or self._default_transforms()
        self.transforms = self._resolve_transforms(selected)

    @property
    def available_transforms(self) -> list[str]:
        return list(self._registry)

    def enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        return self._apply_transforms(df, self.transforms)

    def enrich_for_features(self, df: pl.DataFrame, required_features: set[str]) -> pl.DataFrame:
        transforms = self.transforms_for_features(required_features)
        return self._apply_transforms(df, transforms)

    def transforms_for_features(self, required_features: set[str]) -> list[FeatureTransform]:
        candidates = [
            transform
            for transform in self._registry.values()
            if transform.name in required_features or set(required_features) & transform.output_columns
        ]
        return self._resolve_transforms(candidates)

    def _build_registry(self) -> dict[str, FeatureTransform]:
        transforms: list[FeatureTransform] = [
            VelocityTransform(),
            AccelerationTransform(),
            JerkTransform(),
            EmaStackTransform(periods=self.ema_periods),
            VolumeMaTransform(period=self.volume_ma_period),
            DirectionalMassTransform(volume_ma_period=self.volume_ma_period),
            VpocTransform(lookback=self.vpoc_lookback),
            MarketImpulseTransform(),
        ]
        return {transform.name: transform for transform in transforms}

    def _default_transforms(self) -> tuple[FeatureTransform, ...]:
        return (
            self._registry["velocity"],
            self._registry["acceleration"],
            self._registry["jerk"],
            self._registry["ema_stack"],
            self._registry["volume_ma"],
            self._registry["directional_mass"],
            self._registry["vpoc"],
        )

    def _resolve_transforms(
        self,
        transforms: Sequence[FeatureTransform | str],
    ) -> list[FeatureTransform]:
        resolved: list[FeatureTransform] = []
        seen: set[str] = set()

        def add_transform(item: FeatureTransform | str) -> None:
            transform = self._coerce_transform(item)
            if transform.name in seen:
                return
            for dependency_name in transform.depends_on:
                add_transform(dependency_name)
            resolved.append(transform)
            seen.add(transform.name)

        for item in transforms:
            add_transform(item)
        return resolved

    def _coerce_transform(self, item: FeatureTransform | str) -> FeatureTransform:
        if isinstance(item, str):
            try:
                return self._registry[item]
            except KeyError as exc:
                raise KeyError(f"Unknown transform {item!r}") from exc
        return item

    def _apply_transforms(
        self,
        df: pl.DataFrame,
        transforms: Sequence[FeatureTransform],
    ) -> pl.DataFrame:
        logger.info(
            "Applying Newton transforms: {}",
            ", ".join(transform_names(transforms)) or "(none)",
        )
        result = df
        for transform in transforms:
            missing = set(transform.required_input_columns) - set(result.columns)
            if missing:
                raise ValueError(
                    f"Transform '{transform.name}' requires columns: {sorted(missing)}"
                )
            result = transform.apply(result)
        logger.info("Physics enrichment complete - {} columns total", len(result.columns))
        return result
