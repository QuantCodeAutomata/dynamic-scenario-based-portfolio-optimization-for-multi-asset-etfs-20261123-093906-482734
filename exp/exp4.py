"""Experiment 4: Tail Behavior and Pareto Approximation Analysis of Portfolio
Returns (paper Section 6).

Estimates OLS log-survival and Hill tail-index parameters for the positive
loss and positive gain magnitudes of selected portfolios (Historical/Dynamic
LS TC99, Historical/Dynamic LS MC99, and the buy-and-hold benchmark) at
upper-tail thresholds 5%, 7% and 10%, with 95% stationary-bootstrap
confidence intervals on the OLS estimates.

Run: python -m exp.exp4 [--full]
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import get_baseline_result
from src.config import (
    TAIL_THRESHOLDS, TAIL_BOOTSTRAP_BLOCK_LENGTH, TAIL_BOOTSTRAP_B_PAPER,
    TAIL_BOOTSTRAP_B_RUN, BOOTSTRAP_SEED,
)
from src.tail import tail_indices, ols_tail_index_ci, tail_sample_size, _tail_magnitudes

RESULTS_DIR = "results"

SELECTED = [
    ("Historical LS TC99", ("historical", "LS", "TC99")),
    ("Dynamic LS TC99", ("conditional", "LS", "TC99")),
    ("Historical LS MC99", ("historical", "LS", "MC99")),
    ("Dynamic LS MC99", ("conditional", "LS", "MC99")),
]


def _portfolio_returns(result, key) -> np.ndarray:
    if key == "BHP":
        return result.bhp_returns.to_numpy()
    return result.returns[key].to_numpy()


def main(full: bool = False) -> None:
    n_boot = TAIL_BOOTSTRAP_B_PAPER if full else TAIL_BOOTSTRAP_B_RUN
    print(f"Experiment 4: n_boot={n_boot}, thresholds={TAIL_THRESHOLDS}, full={full}")
    result = get_baseline_result(full=full, progress_every=10)

    portfolios = SELECTED + [("BHP", "BHP")]
    rows = []
    for label, key in portfolios:
        returns = _portfolio_returns(result, key)
        n_total = returns.size
        for threshold in TAIL_THRESHOLDS:
            idx = tail_indices(returns, threshold, n_total)
            row = {"portfolio": label, "threshold": threshold, "n_total": n_total}
            row.update(idx)
            for side in ("loss", "gain"):
                lower, upper = ols_tail_index_ci(
                    returns, threshold, n_total, side,
                    TAIL_BOOTSTRAP_BLOCK_LENGTH, n_boot, BOOTSTRAP_SEED,
                )
                row[f"{side}_ols_ci_lower"] = lower
                row[f"{side}_ols_ci_upper"] = upper
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(f"{RESULTS_DIR}/exp4_tail_indices.csv", index=False)
    print(table.to_string(index=False))

    # Log-survival OLS fit plot at the 7% threshold for Historical/Dynamic LS TC99
    threshold = 0.07
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for label, key in SELECTED[:2]:
        returns = _portfolio_returns(result, key)
        n_total = returns.size
        k = tail_sample_size(threshold, n_total)
        for ax, side in zip(axes, ("loss", "gain")):
            magnitudes = _tail_magnitudes(returns, side)
            k_side = min(k, magnitudes.size - 1)
            order = np.sort(magnitudes)[::-1][:k_side]
            ranks = np.arange(1, k_side + 1)
            survival = ranks / magnitudes.size
            ax.scatter(np.log(order), np.log(survival), s=14, label=f"{label} ({side})")
            slope, intercept = np.polyfit(np.log(order), np.log(survival), 1)
            x_grid = np.log(order)
            ax.plot(x_grid, slope * x_grid + intercept, linewidth=1)
    axes[0].set_title("Loss tail: log-survival vs. log magnitude (7% threshold)")
    axes[1].set_title("Gain tail: log-survival vs. log magnitude (7% threshold)")
    for ax in axes:
        ax.set_xlabel("log positive tail magnitude")
        ax.set_ylabel("log conditional survival probability")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_log_survival_fits.png", dpi=110)
    plt.close(fig)

    summary = table[table.threshold == 0.07][
        ["portfolio", "loss_ols_alpha", "loss_hill_alpha", "gain_ols_alpha", "gain_hill_alpha"]
    ]
    summary.to_csv(f"{RESULTS_DIR}/exp4_summary_7pct.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="paper's full configuration (B=400 OLS bootstrap resamples, 946-session portfolios)")
    args = parser.parse_args()
    main(full=args.full)
