"""Portfolio optimizers: mean-variance, tangent mean-variance, minimum-CVaR,
and tangent-CVaR, all under budget, position-bound, and target-adjustment-cap
constraints (paper Section 3).

Library usage (Context7-checked): `cvxpy` is used as the convex-optimization
DSL for every quadratic (mean-variance) and linear (Rockafellar-Uryasev CVaR)
program, exactly as named in the paper's implementation notes. `pyportfolioopt`
and `skfolio` were also checked; neither exposes the paper's specific
constraint (a turnover cap expressed as (1/2)*||w - w_prev||_1 <= delta
together with a hard position-bound box and, for the CVaR objectives, an
explicit Rockafellar-Uryasev scenario-based linear program), so the objectives
below are assembled directly in cvxpy (paper Section 3, Eqs. for MV/CVaR).
"""

import numpy as np
import cvxpy as cp

from src.config import MV_TOLERANCE, CVAR_MAX_ITER, CVAR_TOLERANCE, CAP_OVERSHOOT_TOL


def _solve_qp_or_lp(problem: cp.Problem, is_qp: bool) -> None:
    # CLARABEL is the installed conic/QP solver in this environment and plays
    # the role of the paper's interior-point solver tolerances (MV_TOLERANCE,
    # CVAR_TOLERANCE); SCS is the fallback when CLARABEL fails to converge.
    tol = MV_TOLERANCE if is_qp else CVAR_TOLERANCE
    max_iter = 20000 if is_qp else CVAR_MAX_ITER
    status = None
    try:
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, max_iter=max_iter)
        status = problem.status
    except cp.error.SolverError:
        status = None
    if status not in ("optimal", "optimal_inaccurate"):
        problem.solve(solver=cp.SCS, max_iters=20000)


def _turnover_constraints(w: cp.Variable, w_prev: np.ndarray, delta: float) -> list:
    """(1/2)||w - w_prev||_1 <= delta (paper Section 3)."""
    return [0.5 * cp.norm1(w - w_prev) <= delta]


def _apply_cap_safety(w: np.ndarray, w_prev: np.ndarray, delta: float) -> np.ndarray:
    """Scale back numerical cap overshoots <= CAP_OVERSHOOT_TOL; the cap is
    never enlarged (paper Section 3)."""
    change = 0.5 * np.abs(w - w_prev).sum()
    if change > delta and change - delta <= CAP_OVERSHOOT_TOL:
        scale = delta / change
        w = w_prev + scale * (w - w_prev)
    return w


def min_variance(cov: np.ndarray, w_prev: np.ndarray, bounds: tuple, delta: float) -> np.ndarray:
    """MV: minimize w'Sigma w s.t. 1'w=1, bounds, turnover cap (paper Section 3)."""
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= bounds[0], w <= bounds[1]]
    constraints += _turnover_constraints(w, w_prev, delta)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
    _solve_qp_or_lp(problem, is_qp=True)
    return _apply_cap_safety(w.value, w_prev, delta)


def _cvar_lp(
    scenarios: np.ndarray,
    alpha: float,
    w_prev: np.ndarray,
    bounds: tuple,
    delta: float,
    target_mean: np.ndarray | None = None,
    target_return: float | None = None,
) -> tuple[np.ndarray, float]:
    """Rockafellar-Uryasev sample-average CVaR minimization (paper Section 3):
    minimize zeta + 1/((1-alpha)*M) * sum(max(loss_s - zeta, 0))
    subject to budget, bounds, turnover cap, and (optionally) a minimum
    expected-return constraint used while tracing the efficient frontier.
    """
    m, n = scenarios.shape
    w = cp.Variable(n)
    zeta = cp.Variable()
    losses = -scenarios @ w
    shortfall = cp.pos(losses - zeta)
    cvar = zeta + cp.sum(shortfall) / ((1.0 - alpha) * m)
    constraints = [cp.sum(w) == 1, w >= bounds[0], w <= bounds[1]]
    constraints += _turnover_constraints(w, w_prev, delta)
    if target_mean is not None and target_return is not None:
        constraints.append(target_mean @ w >= target_return)
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    _solve_qp_or_lp(problem, is_qp=False)
    if w.value is None:
        return None, None
    return _apply_cap_safety(w.value, w_prev, delta), float(cvar.value)


def min_cvar(scenarios: np.ndarray, alpha: float, w_prev: np.ndarray, bounds: tuple, delta: float) -> np.ndarray:
    """MC95/MC99: minimum-CVaR portfolio (paper Section 3)."""
    w, _ = _cvar_lp(scenarios, alpha, w_prev, bounds, delta)
    return w


def tangent_mean_variance(
    mean: np.ndarray,
    cov: np.ndarray,
    rf: float,
    w_prev: np.ndarray,
    bounds: tuple,
    delta: float,
    n_candidates: int,
) -> np.ndarray:
    """TV: maximize (mu'w - r_f)/sqrt(w'Sigma w) by scanning a grid of target
    returns along the turnover-constrained efficient frontier and selecting
    the candidate with the highest realized Sharpe ratio (paper Section 3,
    "toolbox's iterative Sharpe-ratio routine"; implemented here as an
    explicit grid search over `n_candidates` frontier points, the documented
    equivalent of that routine)."""
    n = cov.shape[0]
    lo, hi = mean.min(), mean.max()
    targets = np.linspace(lo, hi, n_candidates)
    best_w, best_sharpe = None, -np.inf
    for target in targets:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= bounds[0], w <= bounds[1], mean @ w >= target]
        constraints += _turnover_constraints(w, w_prev, delta)
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
        _solve_qp_or_lp(problem, is_qp=True)
        if w.value is None:
            continue
        w_val = _apply_cap_safety(w.value, w_prev, delta)
        port_mean = float(mean @ w_val)
        port_vol = float(np.sqrt(max(w_val @ cov @ w_val, 1e-16)))
        sharpe = (port_mean - rf) / port_vol
        if sharpe > best_sharpe:
            best_sharpe, best_w = sharpe, w_val
    assert best_w is not None, "tangent mean-variance frontier search failed to find a feasible point"
    return best_w


def tangent_cvar(
    scenarios: np.ndarray,
    alpha: float,
    rf: float,
    w_prev: np.ndarray,
    bounds: tuple,
    delta: float,
    n_candidates: int,
) -> np.ndarray:
    """TC95/TC99: maximize (mu'w - r_f)/CVaR(w) across the admissible
    turnover-constrained frontier, selecting the candidate with the highest
    excess-return-to-CVaR ratio (paper Section 3)."""
    mean = scenarios.mean(axis=0)
    lo, hi = mean.min(), mean.max()
    targets = np.linspace(lo, hi, n_candidates)
    best_w, best_ratio = None, -np.inf
    for target in targets:
        w, cvar = _cvar_lp(scenarios, alpha, w_prev, bounds, delta, target_mean=mean, target_return=target)
        if w is None or cvar <= 0:
            continue
        port_mean = float(mean @ w)
        ratio = (port_mean - rf) / cvar
        if ratio > best_ratio:
            best_ratio, best_w = ratio, w
    assert best_w is not None, "tangent-CVaR frontier search failed to find a feasible point"
    return best_w


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)
