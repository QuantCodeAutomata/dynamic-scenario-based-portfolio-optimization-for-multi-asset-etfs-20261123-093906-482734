import numpy as np
import pytest

from src.optimization import (
    min_variance, min_cvar, tangent_mean_variance, tangent_cvar, equal_weight,
)


def _synthetic_scenarios(n_assets=5, n_scen=500, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0005, 0.0003, n_assets)
    cov = np.diag(rng.uniform(0.0001, 0.0004, n_assets))
    return rng.multivariate_normal(mean, cov, n_scen)


def test_min_variance_respects_budget_and_bounds():
    scen = _synthetic_scenarios()
    cov = np.cov(scen, rowvar=False)
    w_prev = equal_weight(5)
    w = min_variance(cov, w_prev, bounds=(0.0, 1.0), delta=1.0)  # loose cap
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-6).all() and (w <= 1.0 + 1e-6).all()


def test_turnover_cap_is_respected():
    scen = _synthetic_scenarios()
    cov = np.cov(scen, rowvar=False)
    w_prev = equal_weight(5)
    delta = 1e-4
    w = min_variance(cov, w_prev, bounds=(0.0, 1.0), delta=delta)
    change = 0.5 * np.abs(w - w_prev).sum()
    assert change <= delta + 1e-6


def test_min_cvar_feasible_and_budget_one():
    scen = _synthetic_scenarios()
    w_prev = equal_weight(5)
    w = min_cvar(scen, alpha=0.95, w_prev=w_prev, bounds=(0.0, 1.0), delta=1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-6).all()


def test_long_short_bounds_allow_negative_weights():
    scen = _synthetic_scenarios()
    w_prev = equal_weight(5)
    bounds = (-1.0 / 5, 1.0 + 1.0 / 5)
    w = min_cvar(scen, alpha=0.99, w_prev=w_prev, bounds=bounds, delta=1.0)
    assert w.min() >= bounds[0] - 1e-6
    assert w.max() <= bounds[1] + 1e-6


def test_tangent_mean_variance_beats_min_variance_sharpe():
    scen = _synthetic_scenarios(seed=1)
    mean, cov = scen.mean(axis=0), np.cov(scen, rowvar=False)
    w_prev = equal_weight(5)
    rf = 0.0
    w_tv = tangent_mean_variance(mean, cov, rf, w_prev, bounds=(0.0, 1.0), delta=1.0, n_candidates=10)
    w_mv = min_variance(cov, w_prev, bounds=(0.0, 1.0), delta=1.0)
    sharpe_tv = (mean @ w_tv - rf) / np.sqrt(w_tv @ cov @ w_tv)
    sharpe_mv = (mean @ w_mv - rf) / np.sqrt(w_mv @ cov @ w_mv)
    assert sharpe_tv >= sharpe_mv - 1e-6


def test_tangent_cvar_returns_feasible_weights():
    scen = _synthetic_scenarios(seed=2)
    w_prev = equal_weight(5)
    w = tangent_cvar(scen, alpha=0.95, rf=0.0, w_prev=w_prev, bounds=(0.0, 1.0), delta=1.0, n_candidates=10)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_equal_weight_sums_to_one():
    w = equal_weight(30)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 1.0 / 30)
