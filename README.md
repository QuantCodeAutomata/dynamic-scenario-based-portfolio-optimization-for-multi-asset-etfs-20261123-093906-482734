# Dynamic Scenario-Based Portfolio Optimization for Multi-Asset ETFs

Replication of Mensah, A. A., Jaffri, A. M., Appiah, N., Lindquist, W. B.,
Rachev, S. T., and Fabozzi, F. J. (2026), *Dynamic Scenario-Based Portfolio
Optimization for Multi-Asset ETFs*, arXiv:2611.16540v2 [q-fin.RM].

The paper builds 24 baseline portfolios for 30 U.S.-listed ETFs by combining
two return-input methods (a historical moving-window empirical distribution
and a conditional ARMA(1,1)-GARCH(1,1)-t + Student-t-copula scenario
simulation), two investment-bound regimes (long-only and limited long-short),
and six portfolio objectives (minimum-variance, tangent mean-variance,
minimum-CVaR 95/99, tangent-CVaR 95/99), rebalanced daily over a moving
estimation window with a target-adjustment turnover cap. This repository
reproduces the six selected experiments: the full-sample reward-risk
comparison, the turnover/transaction-cost analysis, the paired bootstrap
inference on historical-vs-conditional differences, the tail-decay analysis,
and two sensitivity analyses (target-adjustment cap; scenario count, seed, and
estimation-window length).

## Layout

- `src/` — shared library: configuration, data-panel construction, the
  ARMA-GARCH-copula scenario engine, the portfolio optimizers, the
  rolling-window backtest engine, reward-risk/turnover/cost metrics, the
  paired stationary-bootstrap inference, and the tail-decay analysis.
- `exp/` — one script per experiment, calling only the `src/` API.
- `data/` — `data_loader.py` loads OHLCV bars for the 30 ETFs and the U.S.
  three-month Treasury par yield; no downloaded data is committed.
- `tests/` — unit tests on synthetic/small fixtures, plus `market_data`-marked
  tests on a small cached real-data slice.
- `results/` — tables, figures, logs and `RESULTS.md` written by the
  experiment scripts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                         # includes market_data tests (needs API keys)
```

## Experiments

Each experiment runs at a reduced configuration by default (see
`results/RESULTS.md`, "Deviations from the paper", for the runtime estimate
that required this) and accepts `--full` to run the paper's full
configuration (946 out-of-sample sessions, S=10,000 scenarios, 20 frontier
candidates, 5,000/400 bootstrap resamples):

```bash
python -m exp.exp1   # historical vs conditional baseline construction and full-sample comparison
python -m exp.exp2   # turnover, implementation, and transaction-cost analysis
python -m exp.exp3   # paired stationary-bootstrap inference
python -m exp.exp4   # tail-decay / Pareto-approximation analysis
python -m exp.exp5   # target-adjustment cap sensitivity (baseline vs 1%)
python -m exp.exp6   # scenario count / seed / estimation-window sensitivity
```

`reproduce.sh` runs the test suite and every experiment in this order.

See `AGENTS.md` for repository conventions and `results/RESULTS.md` for the
realized numbers, their comparison with the paper, and every deviation.
