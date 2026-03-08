"""
Central configuration for the Kinematic Engine.

Uses pydantic-settings to pull from .env and provide typed, validated config.
Gracefully handles missing or inaccessible .env files.
"""

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_env_file() -> str | None:
    """Return the .env path only if it exists and is readable."""
    env_path = PROJECT_ROOT / ".env"
    try:
        if env_path.is_file():
            return str(env_path)
    except (PermissionError, OSError):
        pass
    return None


_ENV_FILE = _resolve_env_file()


class Settings(BaseSettings):
    """Application-wide settings, loaded from .env at project root."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ─────────────────────────────────────────────────────────
    polygon_api_key: str = Field(
        default="",
        description="Polygon.io API key for market data.",
    )

    # ── Data Pipeline Defaults ───────────────────────────────────────────
    default_tickers: List[str] = Field(
        default=["SPY", "QQQ", "IWM"],
        description="Tickers to download by default.",
    )
    lookback_years: int = Field(
        default=2,
        description="Number of years of historical data to fetch.",
    )

    # ── Physics Engine Defaults ──────────────────────────────────────────
    vpoc_lookback_bars: int = Field(
        default=240,
        description="Rolling window size for VPOC (4 hrs × 60 min).",
    )

    # ── Strategy Defaults ────────────────────────────────────────────────
    ema_periods: List[int] = Field(
        default=[4, 8, 12],
        description="EMA periods for the momentum strategy.",
    )
    volume_ma_period: int = Field(
        default=20,
        description="Moving-average window for volume filter.",
    )

    # ── Market Impulse Defaults ────────────────────────────────────────
    vma_length: int = Field(
        default=10,
        description="Lookback for the VMA directional strength.",
    )
    vwma_periods: List[int] = Field(
        default=[8, 21, 34],
        description="VWMA periods for the impulse regime stack.",
    )
    impulse_entry_buffer_minutes: int = Field(
        default=3,
        description="Minutes after open before entries are allowed.",
    )
    impulse_entry_window_minutes: int = Field(
        default=60,
        description="Maximum minutes after open for entries.",
    )
    market_open_hour: int = Field(
        default=9,
        description="Market open hour (ET).",
    )
    market_open_minute: int = Field(
        default=30,
        description="Market open minute (ET).",
    )

    # ── Oracle Defaults ──────────────────────────────────────────────────
    forward_window_bars: int = Field(
        default=15,
        description="Bars to look ahead for MFE / MAE (15 × 1-min = 15 min).",
    )


# Singleton – importable everywhere as `from src.config import settings`
settings = Settings()
