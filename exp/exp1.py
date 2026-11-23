"""Experiment 1: Historical vs Conditional Scenario Portfolio Construction and
Full-Sample Reward-Risk Comparison (paper Sections 2-4).

Constructs the 24 baseline portfolios (2 return-input methods x 2 investment
bounds x 6 objectives) plus the equal-weight buy-and-hold benchmark, then
reproduces: full-sample reward-risk ratios, descriptive moments, kernel
density estimates of realized daily returns, moving-window (overlapping)
risk diagnostics, and gross/net annualized performance.

Run: python -m exp.exp1 [--full]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import get_baseline_result
from src.config import (
    OBJECTIVES, RACHEV_STAR_TAIL_P, RISK_WINDOW_PAPER, RISK_WINDOW_RUN,
    TRANSACTION_COST_RATE, N_EVAL_PAPER, N_EVAL_RUN,
)
from src.metrics import (
    reward_risk_ratios, descriptive_moments, moving_window_diagnostics,
    max_drawdown, annualized_geometric_return, annualized_volatility,
    loss_cvar, implementation_turnover, net_returns,
)

RESULTS_DIR = "results"
BOUNDS = ("LO", "LS")
METHODS = ("historical", "conditional")


def _keys():
    return [(m, b, o) for b in BOUNDS for o in OBJECTIVES for m in METHODS]


def reward_risk_table(result) -> pd.DataFrame:
    rows = []
    for method, bound, objective in _keys():
        gross = result.returns[(method, bound, objective)]
        excess = gross - result.risk_free
        ratios = reward_risk_ratios(excess.to_numpy(), RACHEV_STAR_TAIL_P)
        rows.append({"method": method, "bound": bound, "objective": objective, **ratios})
    bhp_excess = result.bhp_returns - result.risk_free
    rows.append({"method": "BHP", "bound": "-", "objective": "-",
                 **reward_risk_ratios(bhp_excess.to_numpy(), RACHEV_STAR_TAIL_P)})
    return pd.DataFrame(rows)


def moments_table(result) -> pd.DataFrame:
    rows = []
    for method, bound, objective in _keys():
        gross = result.returns[(method, bound, objective)]
        rows.append({"method": method, "bound": bound, "objective": objective,
                     **descriptive_moments(gross.to_numpy())})
    rows.append({"method": "BHP", "bound": "-", "objective": "-",
                 **descriptive_moments(result.bhp_returns.to_numpy())})
    return pd.DataFrame(rows)


def gross_net_table(result) -> pd.DataFrame:
    rows = []
    for method, bound, objective in _keys():
        weights = result.weights[(method, bound, objective)]
        gross = result.returns[(method, bound, objective)]
        kappa = _implementation_turnover_from_result(result, weights)
        # kappa is indexed by decision date t; gross is indexed by the eval
        # date t+1 the turnover's cost is charged to (paper Sections 2, 9).
        # Both share the same row order, so realign kappa onto gross's index.
        kappa_aligned = pd.Series(kappa.to_numpy(), index=gross.index)
        net = net_returns(gross, kappa_aligned, TRANSACTION_COST_RATE)
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
    bhp = result.bhp_returns
    wealth_bhp = 100.0 * (1.0 + bhp).cumprod()
    rows.append({
        "method": "BHP", "bound": "-", "objective": "-",
        "gross_ann_return_pct": annualized_geometric_return(bhp.to_numpy()) * 100,
        "net_ann_return_pct": annualized_geometric_return(bhp.to_numpy()) * 100,
        "gross_ann_vol_pct": annualized_volatility(bhp.to_numpy()) * 100,
        "net_ann_vol_pct": annualized_volatility(bhp.to_numpy()) * 100,
        "gross_cvar95_daily_pct": loss_cvar(bhp.to_numpy()) * 100,
        "net_cvar95_daily_pct": loss_cvar(bhp.to_numpy()) * 100,
        "gross_max_drawdown_pct": max_drawdown(wealth_bhp.to_numpy()),
        "net_max_drawdown_pct": max_drawdown(wealth_bhp.to_numpy()),
    })
    return pd.DataFrame(rows)


def _implementation_turnover_from_result(result, weights: pd.DataFrame) -> pd.Series:
    """kappa_t needs r_t (the return realized over [t-1, t] by the PREVIOUS
    target). ``weights`` is indexed by decision date t; the realized return
    over the same span is ``result``'s eval-date return shifted by one
    position (eval_dates[k] is the return earned between decision_dates[k]
    and decision_dates[k+1])."""
    return implementation_turnover(weights, _asset_return_panel_for(result, weights))


_ASSET_RETURN_CACHE = {}


def _asset_return_panel_for(result, weights: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the per-asset realized-return panel aligned to decision
    dates, needed by ``implementation_turnover``. Loaded once and cached."""
    if "panel" not in _ASSET_RETURN_CACHE:
        from src.backtest import load_price_and_returns
        _, returns, _, _ = load_price_and_returns()
        _ASSET_RETURN_CACHE["panel"] = returns
    returns = _ASSET_RETURN_CACHE["panel"]
    # r_t realized over [t-1, t] for decision date t = the asset return row at
    # the same calendar date as t (returns.index already daily arithmetic
    # returns r_{i,t} = P_t/P_{t-1}-1, so returns.loc[t] IS r_t).
    return returns.loc[weights.index]


def kde_grid(result, path: str) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(10, 18))
    for i, objective in enumerate(OBJECTIVES):
        for j, bound in enumerate(BOUNDS):
            ax = axes[i, j]
            for method, color in (("historical", "tab:blue"), ("conditional", "tab:orange")):
                x = result.returns[(method, bound, objective)].to_numpy()
                grid = np.linspace(x.min(), x.max(), 256)
                density = gaussian_kde(x)(grid)
                ax.plot(grid, density, label=method, color=color)
            ax.set_title(f"{bound} {objective}", fontsize=9)
            ax.set_xlabel("daily return")
            ax.set_ylabel("density")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def moving_window_plot(result, path: str, window: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for method, color in (("historical", "tab:blue"), ("conditional", "tab:orange")):
        gross = result.returns[(method, "LO", "MC95")]
        excess = gross - result.risk_free
        diag = moving_window_diagnostics(gross, excess, window, RACHEV_STAR_TAIL_P)
        axes[0].plot(diag.index, diag["CVaR95"] * 100, label=method, color=color)
        axes[1].plot(diag.index, diag["Sharpe"], label=method, color=color)
    axes[0].set_title(f"Moving-window ({window}-session) CVaR95: LO MC95, historical vs conditional")
    axes[0].set_ylabel("CVaR95 (%)")
    axes[0].legend()
    axes[1].set_title(f"Moving-window ({window}-session) Sharpe: LO MC95, historical vs conditional")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("window end date")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(full: bool = False) -> None:
    n_eval = N_EVAL_PAPER if full else N_EVAL_RUN
    window = RISK_WINDOW_PAPER if full else RISK_WINDOW_RUN
    print(f"Experiment 1: n_eval={n_eval}, moving window={window}, full={full}")
    result = get_baseline_result(full=full, progress_every=10)

    rr = reward_risk_table(result)
    rr.to_csv(f"{RESULTS_DIR}/exp1_reward_risk_ratios.csv", index=False)
    print(rr.to_string(index=False))

    mom = moments_table(result)
    mom.to_csv(f"{RESULTS_DIR}/exp1_descriptive_moments.csv", index=False)

    gn = gross_net_table(result)
    gn.to_csv(f"{RESULTS_DIR}/exp1_gross_net_performance.csv", index=False)

    kde_grid(result, f"{RESULTS_DIR}/exp1_kde_grid.png")
    moving_window_plot(result, f"{RESULTS_DIR}/exp1_moving_window_mc95.png", window)

    n_hist_higher = 0
    n_total = 0
    for bound in BOUNDS:
        for objective in OBJECTIVES:
            for ratio in ("Sharpe", "SS2", "Rachev", "STAR"):
                hist = rr[(rr.method == "historical") & (rr.bound == bound) & (rr.objective == objective)][ratio].iloc[0]
                cond = rr[(rr.method == "conditional") & (rr.bound == bound) & (rr.objective == objective)][ratio].iloc[0]
                n_total += 1
                n_hist_higher += int(hist > cond)
    summary = pd.DataFrame([{"n_comparisons": n_total, "n_historical_higher": n_hist_higher,
                              "share_historical_higher": n_hist_higher / n_total}])
    summary.to_csv(f"{RESULTS_DIR}/exp1_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="paper's full configuration (946 sessions, S=10,000)")
    args = parser.parse_args()
    main(full=args.full)
