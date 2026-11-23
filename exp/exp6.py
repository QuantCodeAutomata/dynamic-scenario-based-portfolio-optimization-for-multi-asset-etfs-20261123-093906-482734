"""Experiment 6: Sensitivity Analysis - Scenario Count, Random Seed, and
Estimation Window Length (paper Section 6).

Three descriptive sub-experiments, all holding the baseline target-adjustment
cap DELTA_BASELINE fixed and comparing annualized geometric returns against a
baseline setting:

  A. Scenario count S in {5,000 / 150, 10,000 / 300, 20,000 / 600} (paper
     value / executed value) for the 12 conditional (dynamic) portfolios.
  B. Simulation random seed in {20261015 (paper baseline), 12345} for the 12
     conditional portfolios.
  C. Estimation window length W in {756, 1008, 1260} for all 24 portfolios,
     evaluated on a common tail of decision dates shared across window
     lengths (paper Section 6).

Run: python -m exp.exp6 [--full]
"""

import argparse

import pandas as pd

from src.backtest import get_scenario_count_result, get_seed_result, get_window_length_result
from src.config import (
    OBJECTIVES, SCENARIO_COUNTS_PAPER, SCENARIO_COUNTS_RUN,
    WINDOW_LENGTHS_PAPER, WINDOW_LENGTHS_RUN, SIMULATION_SEED_PAPER,
    SIMULATION_SEED_ALT, S_PAPER, S_RUN, W_PAPER,
)
from src.metrics import annualized_geometric_return

RESULTS_DIR = "results"
BOUNDS = ("LO", "LS")


def _returns_table(result, method_filter=None) -> dict:
    out = {}
    for bound in BOUNDS:
        for objective in OBJECTIVES:
            for method in ("historical", "conditional"):
                if method_filter is not None and method != method_filter:
                    continue
                key = (method, bound, objective)
                out[key] = annualized_geometric_return(result.returns[key].to_numpy())
    return out


def sub_a_scenario_count(full: bool) -> pd.DataFrame:
    scenario_counts = SCENARIO_COUNTS_PAPER if full else SCENARIO_COUNTS_RUN
    default_s = S_PAPER if full else S_RUN
    rows = []
    baseline_returns = None
    for S in scenario_counts:
        result = get_scenario_count_result(S, full=full, progress_every=5)
        returns = _returns_table(result, method_filter="conditional")
        if S == default_s:
            baseline_returns = returns
        for (method, bound, objective), ann in returns.items():
            rows.append({"S": S, "bound": bound, "objective": objective, "ann_return": ann})
    df = pd.DataFrame(rows)
    df["ann_return_pp_vs_baseline"] = df.apply(
        lambda row: 100 * (row.ann_return - baseline_returns[("conditional", row.bound, row.objective)]),
        axis=1,
    )
    return df


def sub_b_seed(full: bool) -> pd.DataFrame:
    seeds = [SIMULATION_SEED_PAPER, SIMULATION_SEED_ALT]
    rows = []
    baseline_returns = None
    for seed in seeds:
        result = get_seed_result(seed, full=full, progress_every=5)
        returns = _returns_table(result, method_filter="conditional")
        if seed == SIMULATION_SEED_PAPER:
            baseline_returns = returns
        for (method, bound, objective), ann in returns.items():
            rows.append({"seed": seed, "bound": bound, "objective": objective, "ann_return": ann})
    df = pd.DataFrame(rows)
    df["ann_return_pp_vs_baseline"] = df.apply(
        lambda row: 100 * (row.ann_return - baseline_returns[("conditional", row.bound, row.objective)]),
        axis=1,
    )
    return df


def sub_c_window_length(full: bool) -> pd.DataFrame:
    windows = WINDOW_LENGTHS_PAPER if full else WINDOW_LENGTHS_RUN
    rows = []
    baseline_returns = None
    n_eval_by_window = {}
    for W in windows:
        result = get_window_length_result(W, full=full, progress_every=5)
        n_eval_by_window[W] = len(result.eval_dates)
        returns = _returns_table(result)
        if W == W_PAPER:
            baseline_returns = returns
        for (method, bound, objective), ann in returns.items():
            rows.append({"W": W, "method": method, "bound": bound, "objective": objective, "ann_return": ann})
    df = pd.DataFrame(rows)
    df["ann_return_pp_vs_baseline"] = df.apply(
        lambda row: 100 * (row.ann_return - baseline_returns[(row.method, row.bound, row.objective)]),
        axis=1,
    )
    df.attrs["n_eval_by_window"] = n_eval_by_window
    return df


def main(full: bool = False) -> None:
    print(f"Experiment 6: full={full}")

    scen = sub_a_scenario_count(full)
    scen.to_csv(f"{RESULTS_DIR}/exp6_scenario_count_sensitivity.csv", index=False)
    print("Sub-experiment A (scenario count):")
    print(scen.to_string(index=False))

    seed_df = sub_b_seed(full)
    seed_df.to_csv(f"{RESULTS_DIR}/exp6_seed_sensitivity.csv", index=False)
    print("Sub-experiment B (random seed):")
    print(seed_df.to_string(index=False))

    window_df = sub_c_window_length(full)
    window_df.to_csv(f"{RESULTS_DIR}/exp6_window_length_sensitivity.csv", index=False)
    print("Sub-experiment C (estimation window length):")
    print(window_df.to_string(index=False))
    print(f"evaluation dates per window length: {window_df.attrs.get('n_eval_by_window')}")

    summary = pd.DataFrame([{
        "max_abs_pp_scenario_count": scen["ann_return_pp_vs_baseline"].abs().max(),
        "max_abs_pp_seed": seed_df["ann_return_pp_vs_baseline"].abs().max(),
        "max_abs_pp_window_length": window_df["ann_return_pp_vs_baseline"].abs().max(),
    }])
    summary.to_csv(f"{RESULTS_DIR}/exp6_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="paper's full configuration (S in {5000,10000,20000}, 946/694 sessions)")
    args = parser.parse_args()
    main(full=args.full)
