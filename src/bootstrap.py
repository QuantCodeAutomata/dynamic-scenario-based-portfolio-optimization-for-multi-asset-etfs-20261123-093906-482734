"""Paired stationary-bootstrap inference on dynamic-minus-historical
reward-risk-ratio differences, with Holm and Benjamini-Hochberg multiple
testing corrections (paper Section 5/Experiment 3).

Library usage: `arch.bootstrap.StationaryBootstrap` implements the Politis
and Romano (1994) stationary bootstrap; `statsmodels.stats.multitest`
implements the Holm (1979) and Benjamini-Hochberg (1995) adjustments.
"""

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from statsmodels.stats.multitest import multipletests

from src.metrics import reward_risk_ratios


def ratio_difference(
    historical_excess: np.ndarray, conditional_excess: np.ndarray, tail_p: float = 0.05
) -> np.ndarray:
    """Delta = Ratio_dynamic - Ratio_historical for the four reward-risk
    ratios, in the fixed order [Sharpe, SS2, Rachev, STAR] (paper Section
    5)."""
    hist = reward_risk_ratios(historical_excess, tail_p)
    cond = reward_risk_ratios(conditional_excess, tail_p)
    order = ["Sharpe", "SS2", "Rachev", "STAR"]
    return np.array([cond[k] - hist[k] for k in order])


def paired_stationary_bootstrap(
    historical_excess: pd.Series,
    conditional_excess: pd.Series,
    block_length: int,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample common dated indices jointly for the paired historical and
    conditional excess-return series (paper Section 5, step 4) and recompute
    the four reward-risk-ratio differences on each resample.

    Returns (delta_hat, bootstrap_deltas) where ``delta_hat`` has shape (4,)
    and ``bootstrap_deltas`` has shape (n_boot, 4).
    """
    hist = historical_excess.to_numpy()
    cond = conditional_excess.to_numpy()
    assert len(hist) == len(cond), "historical and conditional series must share dated indices"
    delta_hat = ratio_difference(hist, cond)

    sb = StationaryBootstrap(block_length, hist, cond, seed=seed)

    def _stat(h, c):
        return ratio_difference(h, c)

    results = sb.apply(_stat, n_boot)
    return delta_hat, np.asarray(results)


def percentile_ci(bootstrap_deltas: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile confidence interval from the bootstrap distribution of
    Delta* (paper Section 5, step 5)."""
    lower = np.quantile(bootstrap_deltas, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_deltas, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def centered_p_value(delta_hat: np.ndarray, bootstrap_deltas: np.ndarray) -> np.ndarray:
    """Centered two-sided Monte Carlo p-value (paper Section 5, step 6):
    p_hat = (1 + sum_b 1{|Delta*^(b) - Delta_hat| >= |Delta_hat|}) / (B + 1).
    """
    b = bootstrap_deltas.shape[0]
    centered = bootstrap_deltas - delta_hat
    exceed = (np.abs(centered) >= np.abs(delta_hat)).sum(axis=0)
    return (1.0 + exceed) / (b + 1.0)


def adjust_p_values(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Holm (1979) family-wise error rate and Benjamini-Hochberg (1995)
    false-discovery-rate adjustments over one family of raw p-values (paper
    Section 5, step 7)."""
    holm_reject, holm_p, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    bh_reject, bh_p, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return {
        "holm_reject": holm_reject,
        "holm_p": holm_p,
        "bh_reject": bh_reject,
        "bh_p": bh_p,
    }
