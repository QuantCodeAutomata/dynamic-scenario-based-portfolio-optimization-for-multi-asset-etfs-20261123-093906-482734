"""Reward-risk ratios, descriptive moments, tail means, turnover, costs, and
moving-window risk diagnostics (paper Sections 3-4).

All ratios are unannualized daily ratios, computed identically in the
full-sample performance audit and the paired bootstrap (paper Section 4.1).
"""

import numpy as np
import pandas as pd


def tail_mean(y: np.ndarray, p: float) -> float:
    """TM_p(y): exact upper-tail average of y including a fractional cutoff
    observation (paper Section 4.1).

    Custom implementation (paper Eq., Section 4.1): the Acerbi-Tasche exact
    expected-shortfall construction applied to the largest p-fraction of the
    sample. Order y descending y_[1] >= ... >= y_[n]; let k = p*n, k_floor =
    floor(k), and r = k - k_floor the fractional weight of the (k_floor+1)-th
    largest observation. TM_p(y) = (sum_{i=1}^{k_floor} y_[i] + r*y_[k_floor+1]) / k.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    assert n > 0
    k = p * n
    k_floor = int(np.floor(k))
    order = np.sort(y)[::-1]  # descending
    if k_floor >= n:
        return float(order.mean())
    total = order[:k_floor].sum()
    remainder = k - k_floor
    if remainder > 0:
        total += remainder * order[k_floor]
    return float(total / k)


def sharpe_ratio(x: np.ndarray) -> float:
    """Sharpe (1994): mean(x)/std(x), unannualized daily (paper Section 4.1)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x) / np.std(x, ddof=1))


def ss2_ratio(x: np.ndarray) -> float:
    """SS2: mean(max(x,0)) / sqrt(mean(max(-x,0)^2)) (paper Section 4.1)."""
    x = np.asarray(x, dtype=float)
    upside = np.mean(np.maximum(x, 0.0))
    downside = np.sqrt(np.mean(np.maximum(-x, 0.0) ** 2))
    return float(upside / downside)


def rachev_ratio(x: np.ndarray, p: float = 0.05) -> float:
    """Rachev ratio at (1-p)/(1-p): TM_p(x) / TM_p(-x) (paper Section 4.1)."""
    x = np.asarray(x, dtype=float)
    return float(tail_mean(x, p) / tail_mean(-x, p))


def star_ratio(x: np.ndarray, p: float = 0.05) -> float:
    """STAR ratio: mean(x) / TM_p(-x) (paper Section 4.1)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x) / tail_mean(-x, p))


def reward_risk_ratios(excess_returns: np.ndarray, tail_p: float = 0.05) -> dict:
    """The four daily reward-risk ratios used throughout the paper."""
    return {
        "Sharpe": sharpe_ratio(excess_returns),
        "SS2": ss2_ratio(excess_returns),
        "Rachev": rachev_ratio(excess_returns, tail_p),
        "STAR": star_ratio(excess_returns, tail_p),
    }


def descriptive_moments(gross_returns: np.ndarray) -> dict:
    """Mean/std in units of 1e-4, skewness in units of 1e-2, excess kurtosis
    unscaled (paper Section 4.1)."""
    x = np.asarray(gross_returns, dtype=float)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    skew = pd.Series(x).skew()
    kurt = pd.Series(x).kurt()  # pandas reports excess kurtosis (Fisher)
    return {
        "mean_1e4": mean * 1e4,
        "std_1e4": std * 1e4,
        "skew_1e2": skew * 1e2,
        "excess_kurtosis": kurt,
    }


def max_drawdown(wealth: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown on the cumulative wealth path, in
    percent (paper Section 3, "maximum drawdown"; computed on the cumulative
    wealth index since the paper reports wealth paths starting at USD 100)."""
    wealth = np.asarray(wealth, dtype=float)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min() * 100.0)


def annualized_geometric_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """((1+r_1)*...*(1+r_T))^(252/T) - 1 (paper Sections 3-5)."""
    returns = np.asarray(returns, dtype=float)
    t = returns.size
    compounded = np.prod(1.0 + returns)
    return float(compounded ** (periods_per_year / t) - 1.0)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def loss_var(gross_returns: np.ndarray, p: float = 0.05) -> float:
    """95% loss VaR: empirical 95th percentile of negative returns (paper
    Section 4), reported as a positive loss magnitude."""
    losses = -np.asarray(gross_returns, dtype=float)
    return float(np.quantile(losses, 1.0 - p))


def loss_cvar(gross_returns: np.ndarray, p: float = 0.05) -> float:
    """95% loss CVaR: average upper-tail loss with fractional cutoff (paper
    Section 4), reported as a positive loss magnitude, using the exact
    tail-mean construction of ``tail_mean``."""
    losses = -np.asarray(gross_returns, dtype=float)
    return tail_mean(losses, p)


def implementation_turnover(
    weights: pd.DataFrame, realized_returns: pd.DataFrame
) -> pd.Series:
    """Drift-adjusted implementation turnover kappa_t (paper Sections 2, 8-9).

    w_tilde_{i,t} = w_{i,t-1}(1+r_{i,t}) / (1 + w_{t-1}'r_t)
    kappa_t = 0.5 * sum_i |w_{i,t} - w_tilde_{i,t}|

    ``weights`` is indexed by decision date t (the date the target w_t is
    formed) and ``realized_returns`` holds r_t realized over [t-1, t], i.e.
    the return earned by the PREVIOUS target w_{t-1} before today's
    rebalance. The first evaluated date has no previous target so its
    turnover is undefined and is reported as NaN (set to 0 for annualization,
    per paper Section 3's exclusion of the initial formation cost).
    """
    dates = weights.index
    kappa = pd.Series(index=dates, dtype=float)
    prev_w = None
    for t in dates:
        if prev_w is None:
            kappa.loc[t] = np.nan
        else:
            r_t = realized_returns.loc[t]
            wealth_factor = 1.0 + float((prev_w * r_t).sum())
            assert wealth_factor > 0, "non-positive portfolio wealth factor in turnover calc"
            w_tilde = prev_w * (1.0 + r_t) / wealth_factor
            kappa.loc[t] = 0.5 * (weights.loc[t] - w_tilde).abs().sum()
        prev_w = weights.loc[t]
    return kappa


def target_adjustment_turnover(weights: pd.DataFrame) -> pd.Series:
    """tau_t = 0.5*sum_i|w_{i,t} - w_{i,t-1}| (paper Section 2, Table 5)."""
    diffs = weights.diff().abs().sum(axis=1) * 0.5
    diffs.iloc[0] = np.nan
    return diffs


def net_returns(
    gross_portfolio_returns: pd.Series,
    implementation_turnover_series: pd.Series,
    cost_rate: float,
) -> pd.Series:
    """r_net = (1 - c*kappa_t)(1 + w_t'r_{t+1}) - 1 (paper Sections 2, 9)."""
    kappa = implementation_turnover_series.fillna(0.0)
    return (1.0 - cost_rate * kappa) * (1.0 + gross_portfolio_returns) - 1.0


def moving_window_diagnostics(
    gross_returns: pd.Series, excess_returns: pd.Series, window: int, tail_p: float = 0.05
) -> pd.DataFrame:
    """252-session overlapping windows: variance, 95% loss VaR/CVaR (on gross
    returns) and the four reward-risk ratios (on excess returns), paper
    Section 4."""
    rows = []
    n = len(gross_returns)
    for end in range(window, n + 1):
        g = gross_returns.iloc[end - window:end].to_numpy()
        x = excess_returns.iloc[end - window:end].to_numpy()
        row = {
            "date": gross_returns.index[end - 1],
            "variance": float(np.var(g, ddof=1)),
            "VaR95": loss_var(g, tail_p),
            "CVaR95": loss_cvar(g, tail_p),
        }
        row.update(reward_risk_ratios(x, tail_p))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def herfindahl_index(weights: pd.DataFrame) -> pd.Series:
    """HHI = sum of squared (|position| / gross exposure) terms per date
    (paper Section 8)."""
    gross = weights.abs().sum(axis=1)
    normalized = weights.abs().div(gross, axis=0)
    return (normalized ** 2).sum(axis=1)


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().sum(axis=1)


def short_exposure(weights: pd.DataFrame) -> pd.Series:
    return (-weights.clip(upper=0.0)).sum(axis=1)
