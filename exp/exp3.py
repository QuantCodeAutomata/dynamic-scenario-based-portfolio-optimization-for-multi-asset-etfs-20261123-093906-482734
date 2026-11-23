"""Experiment 3: Paired Bootstrap Inference for Historical vs Conditional
Portfolio Differences (paper Section 5).

For each of the 12 matched constraint-objective portfolios and each of the
four reward-risk ratios, computes the dynamic-minus-historical difference
Delta, its paired stationary-bootstrap 95% percentile confidence interval and
centered two-sided p-value at expected block lengths 10, 20 and 40, and
applies Holm and Benjamini-Hochberg corrections across the 48 tests of each
block-length family.

Run: python -m exp.exp3 [--full]
"""

import argparse

import numpy as np
import pandas as pd

from src.backtest import get_baseline_result
from src.config import (
    OBJECTIVES, RACHEV_STAR_TAIL_P, BOOTSTRAP_BLOCK_LENGTHS, BOOTSTRAP_B_PAPER,
    BOOTSTRAP_B_RUN, BOOTSTRAP_SEED, BOOTSTRAP_ALPHA, N_EVAL_PAPER, N_EVAL_RUN,
)
from src.bootstrap import (
    paired_stationary_bootstrap, percentile_ci, centered_p_value, adjust_p_values,
)

RESULTS_DIR = "results"
BOUNDS = ("LO", "LS")
RATIO_NAMES = ["Sharpe", "SS2", "Rachev", "STAR"]


def main(full: bool = False) -> None:
    n_boot = BOOTSTRAP_B_PAPER if full else BOOTSTRAP_B_RUN
    print(f"Experiment 3: n_boot={n_boot}, block lengths={BOOTSTRAP_BLOCK_LENGTHS}, full={full}")
    result = get_baseline_result(full=full, progress_every=10)

    combos = [(bound, objective) for bound in BOUNDS for objective in OBJECTIVES]
    excess = {}
    for bound, objective in combos:
        hist = result.returns[("historical", bound, objective)] - result.risk_free
        cond = result.returns[("conditional", bound, objective)] - result.risk_free
        excess[(bound, objective)] = (hist, cond)

    all_rows = []
    summary_rows = []
    for block_length in BOOTSTRAP_BLOCK_LENGTHS:
        rows = []
        for bound, objective in combos:
            hist_excess, cond_excess = excess[(bound, objective)]
            delta_hat, boot_deltas = paired_stationary_bootstrap(
                hist_excess, cond_excess, block_length, n_boot, BOOTSTRAP_SEED
            )
            lower, upper = percentile_ci(boot_deltas, BOOTSTRAP_ALPHA)
            p_values = centered_p_value(delta_hat, boot_deltas)
            for i, ratio in enumerate(RATIO_NAMES):
                rows.append({
                    "block_length": block_length, "bound": bound, "objective": objective,
                    "ratio": ratio, "delta_hat": delta_hat[i],
                    "ci_lower": lower[i], "ci_upper": upper[i],
                    "ci_excludes_zero": bool(lower[i] > 0 or upper[i] < 0),
                    "p_value": p_values[i],
                })
        block_df = pd.DataFrame(rows)
        adj = adjust_p_values(block_df["p_value"].to_numpy(), BOOTSTRAP_ALPHA)
        block_df["holm_reject"] = adj["holm_reject"]
        block_df["holm_p"] = adj["holm_p"]
        block_df["bh_reject"] = adj["bh_reject"]
        block_df["bh_p"] = adj["bh_p"]
        all_rows.append(block_df)
        summary_rows.append({
            "block_length": block_length,
            "n_tests": len(block_df),
            "n_ci_exclude_zero": int(block_df["ci_excludes_zero"].sum()),
            "n_holm_rejections": int(block_df["holm_reject"].sum()),
            "n_bh_rejections": int(block_df["bh_reject"].sum()),
            "n_delta_negative": int((block_df["delta_hat"] < 0).sum()),
        })
        print(f"block_length={block_length}: "
              f"{summary_rows[-1]['n_ci_exclude_zero']} CIs exclude zero, "
              f"{summary_rows[-1]['n_holm_rejections']} Holm rejections, "
              f"{summary_rows[-1]['n_bh_rejections']} BH rejections")

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(f"{RESULTS_DIR}/exp3_bootstrap_tests.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{RESULTS_DIR}/exp3_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="paper's full configuration (B=5,000 resamples, 946-session portfolios)")
    args = parser.parse_args()
    main(full=args.full)
