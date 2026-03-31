"""Tests for execution stress Monte Carlo."""

import numpy as np

from src.oracle.monte_carlo import ExecutionStressConfig, stress_from_win_flags, stress_profile_library


def test_stress_from_win_flags_shapes_and_probabilities() -> None:
    wins = np.array([True, False, True, True, False, False, True])
    cfg = ExecutionStressConfig(bootstrap_iters=500, random_seed=11)
    out = stress_from_win_flags(wins, ratio=1.5, config=cfg)
    assert out["trades"] == float(len(wins))
    assert 0.0 <= out["mc_prob_positive_exp"] <= 1.0
    assert out["mc_exp_r_p05"] <= out["mc_exp_r_p50"] <= out["mc_exp_r_p95"]


def test_stress_from_empty_returns_zeroes() -> None:
    out = stress_from_win_flags(np.array([], dtype=bool), ratio=2.0, config=ExecutionStressConfig())
    assert out["trades"] == 0
    assert out["mc_prob_positive_exp"] == 0.0


def test_stress_profile_library_includes_named_profiles() -> None:
    profiles = stress_profile_library(bootstrap_iters=250, random_seed=9)
    assert {"default", "debit_spread_tight", "single_option", "stock_like"} <= set(profiles)
    assert profiles["stock_like"].theta_cost_range == (0.0, 0.0)
    assert profiles["default"].bootstrap_iters == 250
