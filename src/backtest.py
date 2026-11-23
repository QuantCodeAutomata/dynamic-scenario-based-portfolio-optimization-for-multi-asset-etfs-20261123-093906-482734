"""Rolling-window portfolio construction and evaluation engine (paper Sections
2-3). Shared by every experiment: the 24 baseline portfolios (2 return-input
methods x 2 investment bounds x 6 objectives), the buy-and-hold benchmark, and
every sensitivity variant (cap, scenario count, seed, window length) are all
produced by calling :func:`run_backtest` with different arguments, never by a
private per-experiment loop.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    BOUNDS_LO, BOUNDS_LS, OBJECTIVES, CVAR_ALPHA, W0, N_ASSETS,
)
from src.copula_scenarios import build_conditional_scenarios
from src.optimization import (
    min_variance, min_cvar, tangent_mean_variance, tangent_cvar, equal_weight,
)

BOUND_MAP = {"LO": BOUNDS_LO, "LS": BOUNDS_LS}
TANGENT_OBJECTIVES = {"TV", "TC95", "TC99"}


@dataclass
class BacktestResult:
    weights: dict  # (method, bound, objective) -> DataFrame (decision_date x ticker)
    returns: dict  # (method, bound, objective) -> Series (realized gross return, indexed by eval date)
    bhp_returns: pd.Series
    risk_free: pd.Series
    eval_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex


def _solve_one(objective: str, mean: np.ndarray, cov: np.ndarray, scenarios: np.ndarray,
                rf: float, w_prev: np.ndarray, bounds: tuple, delta: float,
                frontier_candidates: int) -> np.ndarray:
    if objective == "MV":
        return min_variance(cov, w_prev, bounds, delta)
    if objective == "TV":
        return tangent_mean_variance(mean, cov, rf, w_prev, bounds, delta, frontier_candidates)
    if objective in ("MC95", "MC99"):
        alpha = CVAR_ALPHA[objective]
        return min_cvar(scenarios, alpha, w_prev, bounds, delta)
    if objective in ("TC95", "TC99"):
        alpha = CVAR_ALPHA[objective]
        return tangent_cvar(scenarios, alpha, rf, w_prev, bounds, delta, frontier_candidates)
    raise ValueError(f"unknown objective {objective}")


def buy_and_hold_returns(price_panel: pd.DataFrame, eval_dates: pd.DatetimeIndex) -> pd.Series:
    """Equal-weighted (1/N) buy-and-hold benchmark, no rebalancing, initial
    wealth USD 100 (paper Section 2, benchmark)."""
    start_idx = price_panel.index.get_loc(eval_dates[0]) - 1
    prices = price_panel.iloc[start_idx:].loc[:, :]
    shares = (100.0 * W0) / prices.iloc[0]
    wealth = (prices * shares).sum(axis=1)
    bhp_ret = wealth.pct_change().dropna()
    return bhp_ret.loc[eval_dates]


def run_backtest(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    W: int,
    S: int,
    n_eval: int,
    delta: float,
    seed: int,
    frontier_candidates: int,
    price_panel: pd.DataFrame | None = None,
    bounds: tuple = ("LO", "LS"),
    objectives: tuple = tuple(OBJECTIVES),
    methods: tuple = ("historical", "conditional"),
    progress_every: int = 0,
) -> BacktestResult:
    """Run the rolling-window simulation (paper Sections 2-3).

    At each decision date, the historical return-input is the trailing W-row
    window of ``returns``; the conditional return-input is an S-scenario
    ARMA(1,1)-GARCH(1,1)-t + Student-t-copula simulation fit on the same
    window. Both inputs feed identical constraints (budget, ``bounds``,
    target-adjustment cap ``delta``). The realized portfolio return uses the
    actual next-session ETF returns, never the simulated scenarios.

    The last ``n_eval`` rows of ``returns`` are always the evaluation set, so
    runs with different ``W`` share the same evaluation dates (paper Section
    6/Experiment 6 "common evaluation dates" requirement) as long as
    ``returns`` has at least ``W + n_eval`` rows.
    """
    T = len(returns)
    assert T >= W + n_eval, f"insufficient history: need {W + n_eval} rows, have {T}"
    tickers = returns.columns.tolist()
    rng = np.random.default_rng(seed)

    decision_positions = list(range(T - n_eval - 1, T - 1))
    decision_dates = returns.index[decision_positions]
    eval_dates = returns.index[[p + 1 for p in decision_positions]]

    keys = [(m, b, o) for m in methods for b in bounds for o in objectives]
    w_prev = {key: equal_weight(N_ASSETS) for key in keys}
    weight_rows = {key: [] for key in keys}
    return_rows = {key: [] for key in keys}

    for step, pos in enumerate(decision_positions):
        window = returns.iloc[pos - W + 1: pos + 1]
        assert len(window) == W
        next_ret = returns.iloc[pos + 1].to_numpy()
        rf_t = float(risk_free.loc[decision_dates[step]])

        X_hist = window.to_numpy()
        mean_hist = X_hist.mean(axis=0)
        cov_hist = np.cov(X_hist, rowvar=False)

        if "conditional" in methods:
            scen_cond, _ = build_conditional_scenarios(window, S, rng)
            mean_cond = scen_cond.mean(axis=0)
            cov_cond = np.cov(scen_cond, rowvar=False)
        else:
            scen_cond, mean_cond, cov_cond = None, None, None

        for bound in bounds:
            b = BOUND_MAP[bound]
            for method in methods:
                mean, cov, scen = (
                    (mean_hist, cov_hist, X_hist) if method == "historical"
                    else (mean_cond, cov_cond, scen_cond)
                )
                for objective in objectives:
                    key = (method, bound, objective)
                    w_new = _solve_one(
                        objective, mean, cov, scen, rf_t, w_prev[key], b, delta,
                        frontier_candidates,
                    )
                    w_prev[key] = w_new
                    weight_rows[key].append(w_new)
                    return_rows[key].append(float(w_new @ next_ret))
        if progress_every and (step + 1) % progress_every == 0:
            print(f"  backtest progress: {step + 1}/{len(decision_positions)} decision dates")

    weights_out, returns_out = {}, {}
    for key in keys:
        weights_out[key] = pd.DataFrame(weight_rows[key], index=decision_dates, columns=tickers)
        returns_out[key] = pd.Series(return_rows[key], index=eval_dates, name="_".join(key))

    bhp = buy_and_hold_returns(price_panel, eval_dates) if price_panel is not None else None
    return BacktestResult(
        weights=weights_out,
        returns=returns_out,
        bhp_returns=bhp,
        risk_free=risk_free.loc[eval_dates],
        eval_dates=eval_dates,
        decision_dates=decision_dates,
    )
