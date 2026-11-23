# Repository guide for coding agents and contributors

This repository replicates Mensah, Jaffri, Appiah, Lindquist, Rachev, and
Fabozzi (2026), *Dynamic Scenario-Based Portfolio Optimization for
Multi-Asset ETFs* (arXiv:2611.16540v2): 24 baseline portfolios over 30
U.S.-listed ETFs, built from either a historical moving-window empirical
return matrix or a conditional ARMA(1,1)-GARCH(1,1)-t + Student-t-copula
scenario simulation, under long-only and limited long-short bounds and six
objectives (minimum-variance, tangent mean-variance, minimum-CVaR 95/99,
tangent-CVaR 95/99), rebalanced daily with a target-adjustment turnover cap.

## Layout

- `src/` — flat library modules only: `config.py` (all parameters, each with
  a paper-section/equation source comment), `returns.py`, `garch_copula.py`
  (ARMA-GARCH marginals, kernel CDF, Student-t copula, scenario simulation),
  `optimization.py` (cvxpy mean-variance and CVaR optimizers), `backtest.py`
  (the shared rolling-window engine), `metrics.py` (reward-risk ratios,
  moments, turnover/cost, drawdown), `bootstrap.py` (paired stationary
  bootstrap + Holm/BH correction), `tail.py` (OLS log-survival and Hill tail
  indices with bootstrap CIs).
- `exp/exp1.py` … `exp/exp6.py` — one script per experiment, calling only the
  `src/` functions with their documented signatures and writing all outputs
  directly into `results/`.
- `data/data_loader.py` — loads ETF OHLCV bars through the QuantX Dataloader
  REST API and the U.S. three-month Treasury par yield from FRED; no
  downloaded data is committed (`data/cache/` is git-ignored).
- `tests/` — unit tests on small/synthetic fixtures, plus `market_data`
  -marked tests on a cached reduced real-data slice (skipped automatically
  when `DATALOADER_API_KEY`/`MASSIVE_API_KEY` are absent).
- `results/` — every table (CSV), figure (PNG), and log file an experiment
  script writes, plus `RESULTS.md`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                         # also runs market_data tests (needs API keys)

python -m exp.exp1   # ... python -m exp.exp6     # reduced configuration (default)
python -m exp.exp1 --full   # paper's full configuration (946 sessions, S=10,000, ...)

./reproduce.sh        # tests + all six experiments at the reduced configuration
```

Re-running any `exp.*` command overwrites its files in `results/`; the
numbers and verdicts in `results/RESULTS.md` are written by hand and must be
checked against the newly written files and updated after a rerun.

## Conventions

- Experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables (no classes or dicts), each with a one-line comment
  giving its paper reference and, where a run reduces it for compute time,
  both the `_PAPER` and `_RUN` values side by side.
- Market data is loaded only through `data/data_loader.py`
  (QuantX Dataloader + FRED); `src/` and `exp/` never call a data API
  directly and never invent a private loader.
- `exp/*.py` call the `src/` functions exactly as defined (argument names,
  return dict/DataFrame keys); they do not add undocumented keyword
  arguments or re-implement a shortcut performance calculation.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` via `python-dotenv`; `.env.example` holds only empty
  placeholders.
- Tests: `python -m pytest -m "not market_data"` must stay green without any
  API key; the `market_data` marker is registered in `tests/conftest.py`.
- Every deviation from the paper (data source, reduced configuration,
  unresolved methodological detail) is recorded in `results/RESULTS.md`, not
  in code comments or config defaults.
