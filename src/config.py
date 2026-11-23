"""Module-level configuration for the Dynamic Scenario-Based Portfolio
Optimization for Multi-Asset ETFs replication.

Every constant below is annotated with its meaning, unit, and the paper
section/equation it comes from (or "not specified in the paper" when the
paper leaves it open). Parameters that define the method (window length,
scenario distributional family, risk levels, caps) are never reduced for
runtime; only scalable repetition counts (scenario draws, bootstrap
resamples, frontier grid density, number of out-of-sample sessions) have a
paper value and a reduced, executed value, selected automatically unless the
corresponding experiment is run with --full.
"""

# ---------------------------------------------------------------------------
# Universe (Table A1)
# ---------------------------------------------------------------------------
ETF_TICKERS = [
    "VT", "VTI", "ACWI", "ACWX", "SUSA", "ROUS", "IYG", "BND", "FPE", "BYLD",
    "NTSX", "GAL", "MDIV", "AOM", "AOR", "INKM", "HNDL", "ALTY", "IYLD", "CVY",
    "PDBC", "QAI", "HDG", "MNA", "RAAX", "RLY", "SRET", "YYY", "GAA", "GMOM",
]  # 30 U.S.-listed ETFs, paper Table A1
N_ASSETS = len(ETF_TICKERS)  # N = 30, paper Section 2

# ---------------------------------------------------------------------------
# Sample period (paper Section 2)
# ---------------------------------------------------------------------------
SAMPLE_START = "2018-12-19"  # inclusive, paper Section 2
SAMPLE_END_EXCLUSIVE = "2026-10-01"  # exclusive; last session used is 2026-09-30
IDENTITY_DATE = "2026-09-30"  # point-in-time universe snapshot date for Dataloader

# ---------------------------------------------------------------------------
# Estimation window and decision structure (paper Section 2-3)
# ---------------------------------------------------------------------------
W_PAPER = 1008  # most recent daily returns used to build the return-input, paper Section 2
HOLD = 1  # moving window advances by one observation daily, paper Section 3

# Target-adjustment cap (paper Section 3): (1/2)||w_t - w_{t-1}||_1 <= DELTA
DELTA_BASELINE = 1.0 / (N_ASSETS * 252)  # baseline daily target-adjustment cap
DELTA_RELAXED = 0.01  # relaxed cap used in the Experiment 5 sensitivity analysis

W0 = 1.0 / N_ASSETS  # equal-weight starting allocation, paper Section 3

# Investment bounds (paper Section 3)
BOUNDS_LO = (0.0, 1.0)  # long-only: 0 <= w_i <= 1
BOUNDS_LS = (-1.0 / N_ASSETS, 1.0 + 1.0 / N_ASSETS)  # limited long-short

# Portfolio objectives evaluated at every decision date (paper Section 3)
OBJECTIVES = ["MV", "TV", "MC95", "TC95", "MC99", "TC99"]
CVAR_ALPHA = {"MC95": 0.95, "TC95": 0.95, "MC99": 0.99, "TC99": 0.99}

# Numerical cap overshoot tolerance (paper Section 3): overshoots <=1e-6 scaled back
CAP_OVERSHOOT_TOL = 1e-6

# ---------------------------------------------------------------------------
# Conditional scenario engine: ARMA(1,1)-GARCH(1,1)-t + kernel CDF + Student-t
# copula (paper Section 2)
# ---------------------------------------------------------------------------
S_PAPER = 10_000  # joint pseudo-uniform scenario draws per decision date, paper Section 2
EPSILON_CLIP = 1e-6  # clipping of pseudo-observations to the open unit interval
SIMULATION_SEED_PAPER = 20261015  # fixed before the moving-window loop, paper Section 2
SIMULATION_SEED_ALT = 12345  # alternate seed used only in the Experiment 6 sensitivity check

# Tangent-portfolio frontier search (paper Section 3)
FRONTIER_ILLUSTRATIVE_POINTS = 100  # illustrative 100-point MV/CVaR frontier (no turnover cap)
FRONTIER_CANDIDATES_PAPER = 20  # requested frontier candidates for tangent selection

# ---------------------------------------------------------------------------
# Optimization tolerances (paper Section 3, implementation notes)
# ---------------------------------------------------------------------------
MV_TOLERANCE = 1e-9  # mean-variance optimality / constraint tolerance
CVAR_MAX_ITER = 2000  # CVaR LP maximum iterations
CVAR_TOLERANCE = 1e-8  # CVaR LP absolute/relative gap tolerance

# ---------------------------------------------------------------------------
# Transaction costs and turnover (paper Sections 3-4)
# ---------------------------------------------------------------------------
TRANSACTION_COST_RATE = 0.001  # 10 basis points per unit of one-way implementation turnover

# ---------------------------------------------------------------------------
# Reward-risk ratios and tails (paper Section 4.1)
# ---------------------------------------------------------------------------
RACHEV_STAR_TAIL_P = 0.05  # tail probability used in Rachev 95/95 and STAR 95 (TM_0.05)

# ---------------------------------------------------------------------------
# Moving-window risk diagnostics (paper Section 4)
# ---------------------------------------------------------------------------
RISK_WINDOW = 252  # overlapping moving-window length for variance/VaR/CVaR, paper Section 4
RISK_WINDOW_VAR_CVAR_P = 0.05  # 95% loss VaR/CVaR threshold

# ---------------------------------------------------------------------------
# Paired stationary-bootstrap inference (Experiment 3, paper Section 5)
# ---------------------------------------------------------------------------
BOOTSTRAP_BLOCK_LENGTHS = [10, 20, 40]  # expected block lengths (sessions), paper Section 5
BOOTSTRAP_B_PAPER = 5000  # bootstrap resamples per block length, paper Section 5
BOOTSTRAP_SEED = 12345  # bootstrap random seed, paper Section 5 (distinct from the scenario seed)
BOOTSTRAP_ALPHA = 0.05  # family-wise/FDR significance level

# ---------------------------------------------------------------------------
# Tail analysis (Experiment 4, paper Section 5)
# ---------------------------------------------------------------------------
TAIL_THRESHOLDS = [0.05, 0.07, 0.10]  # upper-tail fractions of the out-of-sample sample
TAIL_BOOTSTRAP_BLOCK_LENGTH = 20  # expected block length for OLS tail-index CI
TAIL_BOOTSTRAP_B_PAPER = 400  # stationary-bootstrap resamples for the OLS tail-index CI

# ---------------------------------------------------------------------------
# Executed (reduced) configuration
# ---------------------------------------------------------------------------
# The paper's full configuration (946 out-of-sample sessions per portfolio,
# S=10,000 scenarios, 20 frontier candidates, 5,000/400 bootstrap resamples)
# requires refitting 30 ARMA-GARCH models, a 30-dimensional Student-t copula,
# and roughly 130 convex programs *per out-of-sample session and per
# experiment*. A timed end-to-end slice (see results/RESULTS.md) showed this
# exceeds the time available for this replication by more than an order of
# magnitude once every experiment and its sensitivity variants are counted.
# Only the scalable repetition counts below are reduced; the estimation
# window length, the decision cadence (HOLD=1), the target-adjustment cap,
# the objective set, and every formula are unchanged, and the full
# configuration remains selectable with each experiment's --full flag.
N_EVAL_PAPER = 946  # paper out-of-sample sessions, Section 2
N_EVAL_RUN = 80  # executed out-of-sample sessions for Experiments 1-4

S_RUN = 300  # executed scenario draws per decision date
FRONTIER_CANDIDATES_RUN = 10  # executed frontier candidates for tangent selection
FRONTIER_ILLUSTRATIVE_RUN = 20  # executed illustrative frontier grid density

BOOTSTRAP_B_RUN = 500  # executed paired-bootstrap resamples (Experiment 3)
TAIL_BOOTSTRAP_B_RUN = 100  # executed OLS tail-index bootstrap resamples (Experiment 4)

# Experiment 5 (target-adjustment cap sensitivity)
N_EVAL_RUN_EXP5 = 40

# Experiment 6 (scenario count / seed / window-length sensitivity)
N_EVAL_RUN_EXP6 = 30
SCENARIO_COUNTS_PAPER = [5_000, 10_000, 20_000]
SCENARIO_COUNTS_RUN = [150, 300, 600]
WINDOW_LENGTHS_PAPER = [756, 1008, 1260]
WINDOW_LENGTHS_RUN = [756, 1008, 1260]  # window length is method-defining: not reduced

# Random seed used for any synthetic/auxiliary randomness outside the paper's
# own fixed seeds (not specified in the paper)
MISC_RANDOM_SEED = 2026
