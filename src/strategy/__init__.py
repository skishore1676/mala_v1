"""Strategy – Configurable logic module."""
from src.strategy.base import BaseStrategy
from src.strategy.ema_momentum import EMAMomentumStrategy

__all__ = ["BaseStrategy", "EMAMomentumStrategy"]
