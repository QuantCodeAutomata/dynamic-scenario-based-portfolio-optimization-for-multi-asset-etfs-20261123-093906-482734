"""Experiment 2: Portfolio Turnover, Implementation, and Transaction Cost
Analysis (paper Sections 2, 8-9).

Separates the constrained target-adjustment turnover from the unconstrained,
drift-adjusted implementation turnover for all 24 baseline portfolios,
reports exposure and concentration measures, and compares gross vs.
transaction-cost-adjusted (net) annualized performance.

Run: python -m exp.exp2 [--full]
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import get_baseline_result, load_price_and_returns
from src.config import (
    OBJECTIVES, TRANSACTION_COST_RATE, N_EVAL_PAPER, N_EVAL_RUN,
)
from src.metrics import (
    target_adjustment_turnover, implementation_turnover, net_returns,
    annualized_geometric_return, annualized_volatility, loss_cvar,
    max_drawdown, herfindahl_index, gross_exposure, short_exposure,
)

RESULTS_DIR = "results"
BOUNDS = ("LO", "LS")
METHODS = ("historical", "conditional")


def _keys():
    return [(m, b, o) for b in BOUNDS for o in OBJECTIVES for m in METHODS]


def turnover_table(result, returns_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, bound, objective in _keys():
        weights = result.weights[(method, bound, objective)]
        gross = result.returns[(method, bound, objective)]

        tau = target_adjustment_turnover(weights)  # indexed by decision date, first = NaN
        tau_ann = tau.dropna().mean() * 252

        kappa = implementation_turnover(weights, returns_panel.loc[weights.index])
        kappa_filled = kappa.fillna(0.0)
        kappa_ann = kappa_filled.mean() * 252

        net = net_returns(gross, pd.Series(kappa_filled.to_numpy(), index=gross.index),
                           TRANSACTION_COST_RATE)
        cost_ann = (annualized_geometric_return(gross.to_numpy())
                    - annualized_geometric_return(net.to_numpy())) * 100

        rows.append({
            "method": method, "bound": bound, "objective": objective,
            "target_adj_turnover_ann_pct": tau_ann * 100,
            "implementation_turnover_ann_pct": kappa_ann * 100,
            "modeled_cost_ann_pct": cost_ann,
            "mean_gross_exposure_pct": gross_exposure(weights).mean() * 100,
            "mean_short_exposure_pct": short_exposure(weights).mean() * 100,
        })
    return pd.DataFrame(rows)


def gross_net_annualized_table(result, returns_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, bound, objective in _keys():
        weights = result.weights[(method, bound, objective)]
        gross = result.returns[(method, bound, objective)]
        kappa = implementation_turnover(weights, returns_panel.loc[weights.index]).fillna(0.0)
        net = net_returns(gross, pd.Series(kappa.to_numpy(), index=gross.index), TRANSACTION_COST_RATE)
        wealth_gross = 100.0 * (1.0 + gross).cumprod()
        wealth_net = 100.0 * (1.0 + net).cumprod()
        rows.append({
            "method": method, "bound": bound, "objective": objective,
            "gross_ann_return_pct": annualized_geometric_return(gross.to_numpy()) * 100,
            "net_ann_return_pct": annualized_geometric_return(net.to_numpy()) * 100,
            "gross_ann_vol_pct": annualized_volatility(gross.to_numpy()) * 100,
            "net_ann_vol_pct": annualized_volatility(net.to_numpy()) * 100,
            "gross_cvar95_daily_pct": loss_cvar(gross.to_numpy()) * 100,
            "net_cvar95_daily_pct": loss_cvar(net.to_numpy()) * 100,
            "gross_max_drawdown_pct": max_drawdown(wealth_gross.to_numpy()),
            "net_max_drawdown_pct": max_drawdown(wealth_net.to_numpy()),
        })
    return pd.DataFrame(rows)


def mc95_concentration_table(result) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for bound in BOUNDS:
            weights = result.weights[(method, bound, "MC95")]
            gross = gross_exposure(weights)
            short = short_exposure(weights)
            hhi = herfindahl_index(weights)
            bond_total = weights[["BND", "BYLD"]].sum(axis=1)
            rows.append({
                "method": method, "bound": bound,
                "BND_weight_mean": weights["BND"].mean(),
                "BYLD_weight_mean": weights["BYLD"].mean(),
                "bond_total_mean": bond_total.mean(),
                "largest_weight_mean": weights.abs().max(axis=1).mean(),
                "HHI_mean": hhi.mean(),
                "gross_exposure_mean_pct": gross.mean() * 100,
                "short_exposure_mean_pct": short.mean() * 100,
            })
    return pd.DataFrame(rows)


def allocation_plot(result, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (method, bound) in zip(
        axes.ravel(), [(m, b) for b in BOUNDS for m in METHODS]
    ):
        weights = result.weights[(method, bound, "MC95")]
        top8 = weights.abs().mean().nlargest(8).index
        for ticker in top8:
            ax.plot(weights.index, weights[ticker], label=ticker)
        ax.set_title(f"{method} {bound} MC95: 8 largest mean |weight| ETFs")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylabel("weight")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def turnover_scatter_plot(turnover_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, marker in (("historical", "o"), ("conditional", "^")):
        sub = turnover_df[turnover_df.method == method]
        ax.scatter(sub["target_adj_turnover_ann_pct"], sub["implementation_turnover_ann_pct"],
                   label=method, marker=marker)
    ax.set_xlabel("annualized target-adjustment turnover (%)")
    ax.set_ylabel("annualized implementation turnover (%)")
    ax.set_title("Target-adjustment vs. implementation turnover, 24 baseline portfolios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(full: bool = False) -> None:
    n_eval = N_EVAL_PAPER if full else N_EVAL_RUN
    print(f"Experiment 2: n_eval={n_eval}, full={full}")
    result = get_baseline_result(full=full, progress_every=10)
    _, returns_panel, _, _ = load_price_and_returns()

    turnover = turnover_table(result, returns_panel)
    turnover.to_csv(f"{RESULTS_DIR}/exp2_turnover.csv", index=False)
    print(turnover.to_string(index=False))

    gross_net = gross_net_annualized_table(result, returns_panel)
    gross_net.to_csv(f"{RESULTS_DIR}/exp2_gross_net_annualized.csv", index=False)

    conc = mc95_concentration_table(result)
    conc.to_csv(f"{RESULTS_DIR}/exp2_mc95_concentration.csv", index=False)
    print(conc.to_string(index=False))

    allocation_plot(result, f"{RESULTS_DIR}/exp2_mc95_allocations.png")
    turnover_scatter_plot(turnover, f"{RESULTS_DIR}/exp2_turnover_scatter.png")

    ls_hist_short = turnover[(turnover.bound == "LS") & (turnover.method == "historical")]["mean_short_exposure_pct"].mean()
    ls_cond_short = turnover[(turnover.bound == "LS") & (turnover.method == "conditional")]["mean_short_exposure_pct"].mean()
    summary = pd.DataFrame([{
        "mean_target_adj_turnover_ann_pct": turnover["target_adj_turnover_ann_pct"].mean(),
        "mean_implementation_turnover_ann_pct": turnover["implementation_turnover_ann_pct"].mean(),
        "mean_ls_historical_short_exposure_pct": ls_hist_short,
        "mean_ls_conditional_short_exposure_pct": ls_cond_short,
    }])
    summary.to_csv(f"{RESULTS_DIR}/exp2_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="paper's full configuration (946 sessions)")
    args = parser.parse_args()
    main(full=args.full)
