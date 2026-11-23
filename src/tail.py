"""Tail-decay analysis of portfolio loss/gain magnitudes: OLS log-survival
regression and the Hill estimator, with stationary-bootstrap confidence
intervals on the OLS tail index (paper Section 6/Experiment 4).

Library usage: `arch.bootstrap.StationaryBootstrap` for the Politis-Romano
resampling used to build the OLS confidence interval; `numpy.polyfit` for the
OLS log-survival regression (a one-feature OLS has no library-specific
estimator beyond ordinary least squares).
"""

import numpy as np
from arch.bootstrap import StationaryBootstrap


def tail_sample_size(threshold: float, n_total: int) -> int:
    """k = ceil(threshold * n_total), the number of upper-order statistics
    entering each fit (paper Section 6: "At the 7% threshold, 67
    observations enter each fit", ceil(0.07*946)=67)."""
    return int(np.ceil(threshold * n_total))


def _tail_magnitudes(returns: np.ndarray, side: str) -> np.ndarray:
    """Positive loss magnitudes (side='loss') or positive gain magnitudes
    (side='gain') drawn from the full return series (paper Section 6, step 2)."""
    returns = np.asarray(returns, dtype=float)
    if side == "loss":
        return -returns[returns < 0]
    return returns[returns > 0]


def ols_tail_index(magnitudes: np.ndarray, k: int) -> float:
    """OLS log-survival regression tail index (paper Section 6, step 4).

    Custom implementation: order the parent magnitude sample descending,
    keep the k largest, assign each the empirical survival probability
    S_i = i / n (n = size of the parent magnitude sample, i = rank from the
    top). Regress log(S_i) on log(magnitude_i) with an intercept; the tail
    index alpha = -slope, since P(X>x) ~ C*x^{-alpha} implies
    log P(X>x) ~ log(C) - alpha*log(x).
    """
    n = magnitudes.size
    assert 0 < k < n, "tail sample size must be smaller than the parent sample"
    order = np.sort(magnitudes)[::-1]
    top = order[:k]
    ranks = np.arange(1, k + 1)
    survival = ranks / n
    slope, _intercept = np.polyfit(np.log(top), np.log(survival), 1)
    return float(-slope)


def hill_estimator(magnitudes: np.ndarray, k: int) -> float:
    """Hill (1975) tail-index estimator using the k largest order statistics
    of the parent magnitude sample (paper Section 6, step 5):
    alpha_Hill = k / (sum_{i=1}^{k} log(X_{(n-i+1)}) - k*log(X_{(n-k)})).
    """
    n = magnitudes.size
    assert 0 < k < n, "tail sample size must be smaller than the parent sample"
    ascending = np.sort(magnitudes)
    top_k = ascending[n - k:]  # the k largest, ascending order
    threshold = ascending[n - k - 1]
    return float(k / (np.sum(np.log(top_k)) - k * np.log(threshold)))


def tail_indices(returns: np.ndarray, threshold: float, n_total: int) -> dict:
    """OLS and Hill tail indices for both the loss and gain tails at one
    threshold (paper Section 6)."""
    k = tail_sample_size(threshold, n_total)
    out = {}
    for side in ("loss", "gain"):
        magnitudes = _tail_magnitudes(returns, side)
        k_side = min(k, magnitudes.size - 1)
        out[f"{side}_ols_alpha"] = ols_tail_index(magnitudes, k_side)
        out[f"{side}_hill_alpha"] = hill_estimator(magnitudes, k_side)
        out[f"{side}_n_obs"] = k_side
    return out


def ols_tail_index_ci(
    returns: np.ndarray,
    threshold: float,
    n_total: int,
    side: str,
    block_length: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% stationary-bootstrap percentile confidence interval for the OLS
    tail-index estimate (paper Section 6, step 6)."""
    returns = np.asarray(returns, dtype=float)

    def _stat(sample: np.ndarray) -> float:
        magnitudes = _tail_magnitudes(sample, side)
        k = tail_sample_size(threshold, n_total)
        k = min(k, max(magnitudes.size - 1, 1))
        if magnitudes.size < 5:
            return np.nan
        return ols_tail_index(magnitudes, k)

    sb = StationaryBootstrap(block_length, returns, seed=seed)
    draws = sb.apply(_stat, n_boot).ravel()
    draws = draws[~np.isnan(draws)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
