"""Experiment 5: Sensitivity Analysis - Target-Adjustment Cap Variation
(paper Section 6).

Reconstructs all 24 baseline portfolios with the daily target-adjustment cap
relaxed from delta=1/(30*252) to delta=1%, holding every other setting fixed,
and compares implementation turnover, modeled transaction costs, gross and
short exposure, and gross/net annualized geometric returns.

Run: python -m exp.exp5 [--full]
"""

import argparse

import numpy as np
import pandas as pd

from src.backtest import get_cap_sensitivity_result, load_price_and_returns
from src.config import OBJECTIVES, TRANSACTION_COST_RATE
from src.metrics import (
    implementation_turnover, net_returns, annualized_geometric_return,
    gross_exposure, short_exposure,
)

RESULTS_DIR = "results"
BOUNDS = ("LO", "LS")

_ASSET_RETURN_CACHE = {}


def _asset_return_panel_for(weights: pd.DataFrame) -> pd.DataFrame:
    """Realized per-asset arithmetic-return panel aligned to decision dates,
    needed by ``implementation_turnover`` (paper Sections 2, 9); loaded once
    and cached across portfolios."""
    if "panel" not in _ASSET_RETURN_CACHE:
        _, returns, _, _ = load_price_and_returns()
        _ASSET_RETURN_CACHE["panel"] = returns
    return _ASSET_RETURN_CACHE["panel"].loc[weights.index]


def _portfolio_table(result, label: str) -> pd.DataFrame:
    rows = []
    for bound in BOUNDS:
        for objective in OBJECTIVES:
            for method in ("historical", "conditional"):
                key = (method, bound, objective)
                weights = result.weights[key]
                gross = result.returns[key]
                kappa = implementation_turnover(weights, _asset_return_panel_for(weights))
                kappa_aligned = pd.Series(kappa.to_numpy(), index=gross.index)
                net = net_returns(gross, kappa_aligned, TRANSACTION_COST_RATE)
                turnover_annualized = kappa.iloc[1:].mean() * 252
                rows.append({
                    "cap": label, "method": method, "bound": bound, "objective": objective,
                    "implementation_turnover_annualized": turnover_annualized,
                    "modeled_cost_annualized": turnover_annualized * TRANSACTION_COST_RATE,
                    "mean_gross_exposure": gross_exposure(weights).mean(),
                    "mean_short_exposure": short_exposure(weights).mean(),
                    "gross_annualized_return": annualized_geometric_return(gross.to_numpy()),
                    "net_annualized_return": annualized_geometric_return(net.to_numpy()),
                })
    return pd.DataFrame(rows)


def main(full: bool = False) -> None:
    print(f"Experiment 5: full={full}")
    baseline = get_cap_sensitivity_result(relaxed=False, full=full, progress_every=5)
    relaxed = get_cap_sensitivity_result(relaxed=True, full=full, progress_every=5)
    print(f"baseline eval dates: {len(baseline.eval_dates)}, relaxed eval dates: {len(relaxed.eval_dates)}")

    base_table = _portfolio_table(baseline, "baseline_delta")
    relaxed_table = _portfolio_table(relaxed, "relaxed_1pct")
    combined = pd.concat([base_table, relaxed_table], ignore_index=True)
    combined.to_csv(f"{RESULTS_DIR}/exp5_cap_sensitivity.csv", index=False)

    merged = base_table.merge(
        relaxed_table, on=["method", "bound", "objective"], suffixes=("_baseline", "_relaxed")
    )
    merged["delta_gross_pp"] = 100 * (merged["gross_annualized_return_relaxed"] - merged["gross_annualized_return_baseline"])
    merged["delta_net_pp"] = 100 * (merged["net_annualized_return_relaxed"] - merged["net_annualized_return_baseline"])
    merged.to_csv(f"{RESULTS_DIR}/exp5_cap_comparison.csv", index=False)

    mc95 = merged[merged.objective == "MC95"][
        ["method", "bound", "implementation_turnover_annualized_baseline",
         "implementation_turnover_annualized_relaxed", "modeled_cost_annualized_baseline",
         "modeled_cost_annualized_relaxed", "mean_gross_exposure_baseline", "mean_gross_exposure_relaxed"]
    ]
    mc95.to_csv(f"{RESULTS_DIR}/exp5_mc95_focus.csv", index=False)
    print(mc95.to_string(index=False))
    print(merged[["method", "bound", "objective", "delta_gross_pp", "delta_net_pp"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="paper's full configuration (946-session portfolios)")
    args = parser.parse_args()
    main(full=args.full)
