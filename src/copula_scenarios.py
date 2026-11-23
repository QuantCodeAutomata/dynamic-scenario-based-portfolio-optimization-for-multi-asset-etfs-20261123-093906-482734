"""Conditional scenario engine: ARMA(1,1)-GARCH(1,1)-t marginals, kernel CDF
transforms, a multivariate Student-t copula, and one-step-ahead scenario
return generation (paper Section 2).

Library usage:
- `arch.arch_model` (package `arch`) fits the ARMA(1,1)-GARCH(1,1) model with
  Student-t innovations (paper Section 2).
- `scipy.stats.gaussian_kde` builds the kernel CDF/inverse-CDF of the
  standardized residuals (paper's MATLAB `ksdensity` equivalent).
- The multivariate Student-t copula itself (fit degrees of freedom and
  correlation matrix by approximate maximum likelihood, and random sampling)
  has no ready-made, dependency-light Python equivalent compatible with this
  environment's pinned numpy/scipy versions (the `copulae` package requires
  numpy>=2, which breaks `arch`/`scikit-learn` ABI compatibility here), so it
  is implemented directly from `scipy.stats.multivariate_t` and a profile
  likelihood over the degrees-of-freedom parameter, following the
  correlation-matrix/degrees-of-freedom factorization used by MATLAB's
  `copulafit('t', ...)` (paper Section 2).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.tsa.arima.model import ARIMA

from src.config import EPSILON_CLIP


@dataclass
class MarginalFit:
    mu_hat: float  # one-step-ahead conditional mean forecast
    sigma_hat: float  # one-step-ahead conditional volatility forecast
    z_hat: np.ndarray  # in-sample standardized residuals


def fit_marginal(returns: np.ndarray) -> MarginalFit:
    """Fit ARMA(1,1)-GARCH(1,1) with Student-t innovations to one ETF's
    return window and return the one-step-ahead mean/volatility forecast and
    standardized residuals (paper Section 2).

    `arch.arch_model`'s built-in mean specifications ('Constant', 'AR',
    'ARX', 'HAR', 'HARX') do not include a moving-average term, so there is
    no single-call joint ARMA(1,1)-GARCH(1,1) estimator in the `arch`
    package. This follows the standard two-step approximation: the ARMA(1,1)
    conditional-mean equation mu_hat = delta_0 + delta_1*r_t + theta*epsilon_t
    is fit with `statsmodels.tsa.arima.model.ARIMA(order=(1,0,1))`, and the
    GARCH(1,1)-t volatility equation sigma_hat^2 = omega + alpha*epsilon_t^2 +
    beta*sigma_t^2 is then fit on the ARMA residuals with a zero-mean
    `arch_model` (paper Section 2, implementation notes).
    """
    scaled = pd.Series(returns * 100.0)
    arma = ARIMA(scaled, order=(1, 0, 1), trend="c").fit()
    mu_hat = float(arma.forecast(steps=1).iloc[0]) / 100.0
    resid = arma.resid.to_numpy() if hasattr(arma.resid, "to_numpy") else arma.resid

    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    garch_res = garch.fit(disp="off", show_warning=False)
    forecast = garch_res.forecast(horizon=1, reindex=False)
    sigma_hat = float(np.sqrt(forecast.variance.values[-1, 0])) / 100.0
    std_resid = garch_res.std_resid
    z_hat = std_resid.to_numpy() if hasattr(std_resid, "to_numpy") else np.asarray(std_resid)
    z_hat = z_hat[~np.isnan(z_hat)]
    return MarginalFit(mu_hat=mu_hat, sigma_hat=sigma_hat, z_hat=z_hat)


class KernelTransform:
    """Kernel-based CDF and inverse CDF of standardized residuals
    (scipy.stats.gaussian_kde), used to map residuals to pseudo-uniforms and
    back (paper Section 2)."""

    def __init__(self, z: np.ndarray, grid_size: int = 2000):
        self.kde = stats.gaussian_kde(z)
        lo, hi = z.min() - 5 * z.std(), z.max() + 5 * z.std()
        self.grid = np.linspace(lo, hi, grid_size)
        pdf = self.kde(self.grid)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        self.cdf_grid = cdf

    def cdf(self, z: np.ndarray) -> np.ndarray:
        return np.interp(z, self.grid, self.cdf_grid)

    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        return np.interp(u, self.cdf_grid, self.grid)


def clip_unit_interval(u: np.ndarray, epsilon: float = EPSILON_CLIP) -> np.ndarray:
    """u* = min(1-epsilon, max(epsilon, u)) (paper Section 2)."""
    return np.minimum(1.0 - epsilon, np.maximum(epsilon, u))


def _normal_score(u: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(u)


def fit_student_t_copula(pseudo_obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Approximate maximum likelihood fit of a multivariate Student-t copula
    to a (W, N) matrix of clipped pseudo-observations in (0,1).

    Follows the two-step approximate-MLE approach used by MATLAB's
    copulafit('t', ...): the correlation matrix is estimated from Kendall's
    tau (robust, scale-free), and the degrees-of-freedom parameter is then
    chosen to maximize the copula log-likelihood by a 1-D profile search
    (custom implementation, paper Section 2, "approximate maximum
    likelihood").
    """
    n, dim = pseudo_obs.shape
    tau = np.ones((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            t, _ = stats.kendalltau(pseudo_obs[:, i], pseudo_obs[:, j])
            tau[i, j] = tau[j, i] = t
    corr = np.sin(np.pi / 2.0 * tau)  # Kendall's-tau-to-Pearson-correlation map for elliptical copulas
    # ensure positive semi-definite (numerical safeguard)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    z = _normal_score(pseudo_obs)

    def neg_log_likelihood(log_nu: float) -> float:
        nu = np.exp(log_nu)
        t_scores = stats.t.ppf(pseudo_obs, df=nu)
        try:
            mvt_logpdf = stats.multivariate_t.logpdf(t_scores, loc=np.zeros(dim), shape=corr, df=nu)
        except Exception:
            return 1e12
        marginal_logpdf = stats.t.logpdf(t_scores, df=nu).sum(axis=1)
        copula_loglik = mvt_logpdf - marginal_logpdf
        if not np.all(np.isfinite(copula_loglik)):
            return 1e12
        return -copula_loglik.sum()

    result = minimize_scalar(neg_log_likelihood, bounds=(np.log(2.5), np.log(60.0)), method="bounded")
    nu_hat = float(np.exp(result.x))
    return corr, nu_hat


def sample_student_t_copula(corr: np.ndarray, nu: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_samples`` joint pseudo-uniform vectors from the fitted
    Student-t copula (paper Section 2)."""
    dim = corr.shape[0]
    mvt = stats.multivariate_t(loc=np.zeros(dim), shape=corr, df=nu, seed=rng)
    draws = mvt.rvs(size=n_samples)
    return stats.t.cdf(draws, df=nu)


@dataclass
class ConditionalFit:
    """One day's fitted ARMA-GARCH marginals and Student-t copula, cached so
    that the sensitivity analyses over scenario count and random seed
    (paper Section 6/Experiment 6) can resample the SAME fitted model instead
    of refitting it, matching the paper's own description that "the same
    marginal fits and copula parameters are estimated; only the number of
    draws" (scenario count) or "the random draws" (seed) differ."""

    transforms: list
    mus: np.ndarray
    sigmas: np.ndarray
    corr: np.ndarray
    nu: float


def fit_conditional_model(window_returns: pd.DataFrame) -> ConditionalFit:
    """Fit the ARMA-GARCH-t marginals and the Student-t copula on one
    estimation window, without drawing any scenarios yet (paper Section 2,
    steps 2-3)."""
    n_assets = window_returns.shape[1]
    z_hats, mus, sigmas, transforms = [], [], [], []
    min_len = None
    for col in window_returns.columns:
        fit = fit_marginal(window_returns[col].to_numpy())
        transform = KernelTransform(fit.z_hat)
        z_hats.append(fit.z_hat)
        mus.append(fit.mu_hat)
        sigmas.append(fit.sigma_hat)
        transforms.append(transform)
        min_len = fit.z_hat.size if min_len is None else min(min_len, fit.z_hat.size)

    pseudo_obs = np.column_stack([
        clip_unit_interval(transforms[i].cdf(z_hats[i][-min_len:])) for i in range(n_assets)
    ])
    corr, nu = fit_student_t_copula(pseudo_obs)
    return ConditionalFit(transforms=transforms, mus=np.array(mus), sigmas=np.array(sigmas), corr=corr, nu=nu)


def sample_conditional_scenarios(
    fit: ConditionalFit, n_scenarios: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_scenarios`` one-step-ahead scenario returns from an already
    fitted conditional model (paper Section 2, step 4)."""
    n_assets = len(fit.transforms)
    u_scenarios = clip_unit_interval(sample_student_t_copula(fit.corr, fit.nu, n_scenarios, rng))
    scenario_returns = np.empty((n_scenarios, n_assets))
    for i in range(n_assets):
        z_sim = fit.transforms[i].inverse_cdf(u_scenarios[:, i])  # NOT recentered or rescaled
        scenario_returns[:, i] = fit.mus[i] + fit.sigmas[i] * z_sim
    return scenario_returns


def build_conditional_scenarios(
    window_returns: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead conditional scenario returns for all assets in
    ``window_returns`` (paper Section 2, steps 3-4).

    Returns (scenarios, moments) where ``scenarios`` has shape
    (n_scenarios, n_assets) and ``moments`` has shape (n_assets, 2) holding
    (mu_hat, sigma_hat) for diagnostics.
    """
    fit = fit_conditional_model(window_returns)
    scenario_returns = sample_conditional_scenarios(fit, n_scenarios, rng)
    moments = np.column_stack([fit.mus, fit.sigmas])
    return scenario_returns, moments
