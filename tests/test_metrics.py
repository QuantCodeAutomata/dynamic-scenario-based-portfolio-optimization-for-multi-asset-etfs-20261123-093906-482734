import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    tail_mean, sharpe_ratio, ss2_ratio, rachev_ratio, star_ratio,
    descriptive_moments, max_drawdown, annualized_geometric_return,
    annualized_volatility, loss_var, loss_cvar, implementation_turnover,
    target_adjustment_turnover, net_returns, moving_window_diagnostics,
    herfindahl_index, gross_exposure, short_exposure,
)


def test_tail_mean_matches_full_mean_when_p_equals_one():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert tail_mean(y, 1.0) == pytest.approx(y.mean())


def test_tail_mean_fractional_cutoff():
    # n=10, p=0.25 -> k=2.5, k_floor=2, remainder=0.5
    y = np.arange(1, 11, dtype=float)  # 1..10
    order = np.sort(y)[::-1]  # 10,9,8,...
    expected = (order[:2].sum() + 0.5 * order[2]) / 2.5
    assert tail_mean(y, 0.25) == pytest.approx(expected)


def test_tail_mean_single_observation():
    assert tail_mean(np.array([5.0]), 1.0) == pytest.approx(5.0)


def test_sharpe_ratio_sign_and_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 2000)
    assert sharpe_ratio(x) > 0
    assert sharpe_ratio(-x) < 0


def test_ss2_ratio_positive_for_positive_mean_series():
    rng = np.random.default_rng(1)
    x = rng.normal(0.002, 0.01, 2000)
    assert ss2_ratio(x) > 0


def test_rachev_and_star_ratios_finite():
    rng = np.random.default_rng(2)
    x = rng.standard_t(5, 1000) * 0.01
    r = rachev_ratio(x, 0.05)
    s = star_ratio(x, 0.05)
    assert np.isfinite(r) and np.isfinite(s)


def test_descriptive_moments_units():
    x = np.array([0.001, -0.002, 0.0015, -0.0005, 0.0003])
    moments = descriptive_moments(x)
    assert moments["mean_1e4"] == pytest.approx(np.mean(x) * 1e4)
    assert moments["std_1e4"] == pytest.approx(np.std(x, ddof=1) * 1e4)


def test_max_drawdown_known_path():
    wealth = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
    # trough 90 after peak 110 -> drawdown = 90/110-1
    assert max_drawdown(wealth) == pytest.approx((90.0 / 110.0 - 1.0) * 100.0)


def test_annualized_geometric_return_known_value():
    returns = np.full(252, 0.0)
    assert annualized_geometric_return(returns, 252) == pytest.approx(0.0)
    returns_up = np.full(252, (1.1 ** (1 / 252)) - 1.0)
    assert annualized_geometric_return(returns_up, 252) == pytest.approx(0.1, abs=1e-6)


def test_annualized_volatility_scaling():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 1000)
    assert annualized_volatility(x) == pytest.approx(np.std(x, ddof=1) * np.sqrt(252))


def test_loss_var_and_cvar_relationship():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, 2000)
    var95 = loss_var(x, 0.05)
    cvar95 = loss_cvar(x, 0.05)
    assert cvar95 >= var95  # CVaR averages tail beyond VaR, so it is at least as large


def test_target_adjustment_turnover_first_row_nan_and_nonnegative():
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.6, 0.4], [0.6, 0.4]], index=pd.date_range("2024-01-01", periods=3)
    )
    tau = target_adjustment_turnover(weights)
    assert np.isnan(tau.iloc[0])
    assert tau.iloc[1] == pytest.approx(0.1)
    assert tau.iloc[2] == pytest.approx(0.0)


def test_implementation_turnover_zero_when_no_drift_and_no_rebalance():
    dates = pd.date_range("2024-01-01", periods=3)
    weights = pd.DataFrame([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], index=dates)
    realized = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], index=dates)
    kappa = implementation_turnover(weights, realized)
    assert np.isnan(kappa.iloc[0])
    assert kappa.iloc[1] == pytest.approx(0.0)
    assert kappa.iloc[2] == pytest.approx(0.0)


def test_net_returns_below_gross_returns_when_turnover_positive():
    gross = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    kappa = pd.Series([0.1, 0.2], index=gross.index)
    net = net_returns(gross, kappa, cost_rate=0.001)
    assert (net < gross).all()


def test_moving_window_diagnostics_shape():
    dates = pd.date_range("2024-01-01", periods=30)
    gross = pd.Series(np.random.default_rng(5).normal(0, 0.01, 30), index=dates)
    excess = gross - 0.00005
    diag = moving_window_diagnostics(gross, excess, window=10)
    assert len(diag) == 30 - 10 + 1
    assert {"variance", "VaR95", "CVaR95", "Sharpe"}.issubset(diag.columns)


def test_herfindahl_index_bounds():
    weights = pd.DataFrame([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25]])
    hhi = herfindahl_index(weights)
    assert hhi.iloc[0] == pytest.approx(1.0)
    assert 1.0 / 3.0 <= hhi.iloc[1] <= 1.0


def test_gross_and_short_exposure():
    weights = pd.DataFrame([[1.2, -0.2], [0.5, 0.5]])
    assert gross_exposure(weights).iloc[0] == pytest.approx(1.4)
    assert short_exposure(weights).iloc[0] == pytest.approx(0.2)
    assert short_exposure(weights).iloc[1] == pytest.approx(0.0)
