"""Monte Carlo stress testing utilities for strategy execution assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionStressConfig:
    bootstrap_iters: int = 5000
    win_fill_range: tuple[float, float] = (0.70, 1.00)
    loss_slip_range: tuple[float, float] = (1.00, 1.30)
    theta_cost_range: tuple[float, float] = (0.02, 0.10)
    fee_cost_range: tuple[float, float] = (0.01, 0.04)
    capture_mult_range: tuple[float, float] = (0.85, 1.20)
    random_seed: int = 7


def stress_profile_library(*, bootstrap_iters: int = 5000, random_seed: int = 7) -> dict[str, ExecutionStressConfig]:
    return {
        "default": ExecutionStressConfig(
            bootstrap_iters=bootstrap_iters,
            random_seed=random_seed,
        ),
        "debit_spread_tight": ExecutionStressConfig(
            bootstrap_iters=bootstrap_iters,
            win_fill_range=(0.82, 1.00),
            loss_slip_range=(1.00, 1.18),
            theta_cost_range=(0.01, 0.06),
            fee_cost_range=(0.01, 0.03),
            capture_mult_range=(0.90, 1.12),
            random_seed=random_seed,
        ),
        "single_option": ExecutionStressConfig(
            bootstrap_iters=bootstrap_iters,
            win_fill_range=(0.78, 1.00),
            loss_slip_range=(1.00, 1.22),
            theta_cost_range=(0.015, 0.08),
            fee_cost_range=(0.01, 0.03),
            capture_mult_range=(0.88, 1.18),
            random_seed=random_seed,
        ),
        "stock_like": ExecutionStressConfig(
            bootstrap_iters=bootstrap_iters,
            win_fill_range=(0.97, 1.00),
            loss_slip_range=(1.00, 1.03),
            theta_cost_range=(0.0, 0.0),
            fee_cost_range=(0.0, 0.01),
            capture_mult_range=(0.98, 1.02),
            random_seed=random_seed,
        ),
    }


def stress_from_win_flags(
    win_flags: np.ndarray,
    ratio: float,
    config: ExecutionStressConfig,
) -> dict[str, float]:
    """
    Simulate execution-stressed expectancy distribution from binary win/loss events.

    Base per-trade R:
      - win  -> +ratio
      - loss -> -1.0
    Then perturb with randomized fill/cost assumptions.
    """
    flags = np.asarray(win_flags, dtype=bool)
    n = len(flags)
    if n == 0:
        return {
            "trades": 0,
            "mc_exp_r_mean": 0.0,
            "mc_exp_r_p05": 0.0,
            "mc_exp_r_p50": 0.0,
            "mc_exp_r_p95": 0.0,
            "mc_prob_positive_exp": 0.0,
            "mc_total_r_p05": 0.0,
            "mc_total_r_p50": 0.0,
            "mc_total_r_p95": 0.0,
            "mc_max_dd_p50": 0.0,
        }

    rng = np.random.default_rng(config.random_seed)
    iters = int(config.bootstrap_iters)

    # Bootstrap trades with replacement for each iteration.
    idx = rng.integers(0, n, size=(iters, n))
    wins = flags[idx]
    base_r = np.where(wins, ratio, -1.0).astype(np.float64)

    # Stress factors.
    capture = rng.uniform(
        config.capture_mult_range[0],
        config.capture_mult_range[1],
        size=(iters, n),
    )
    win_fill = rng.uniform(
        config.win_fill_range[0],
        config.win_fill_range[1],
        size=(iters, n),
    )
    loss_slip = rng.uniform(
        config.loss_slip_range[0],
        config.loss_slip_range[1],
        size=(iters, n),
    )
    theta = rng.uniform(
        config.theta_cost_range[0],
        config.theta_cost_range[1],
        size=(iters, n),
    )
    fees = rng.uniform(
        config.fee_cost_range[0],
        config.fee_cost_range[1],
        size=(iters, n),
    )

    stressed = base_r * capture
    stressed = np.where(wins, stressed * win_fill, stressed * loss_slip)
    stressed = stressed - theta - fees

    exp_r = stressed.mean(axis=1)
    total_r = stressed.sum(axis=1)
    equity = np.cumsum(stressed, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    max_dd = (peak - equity).max(axis=1)

    return {
        "trades": float(n),
        "mc_exp_r_mean": float(np.mean(exp_r)),
        "mc_exp_r_p05": float(np.quantile(exp_r, 0.05)),
        "mc_exp_r_p50": float(np.quantile(exp_r, 0.50)),
        "mc_exp_r_p95": float(np.quantile(exp_r, 0.95)),
        "mc_prob_positive_exp": float(np.mean(exp_r > 0.0)),
        "mc_total_r_p05": float(np.quantile(total_r, 0.05)),
        "mc_total_r_p50": float(np.quantile(total_r, 0.50)),
        "mc_total_r_p95": float(np.quantile(total_r, 0.95)),
        "mc_max_dd_p50": float(np.quantile(max_dd, 0.50)),
    }


__all__ = ["ExecutionStressConfig", "stress_from_win_flags", "stress_profile_library"]
