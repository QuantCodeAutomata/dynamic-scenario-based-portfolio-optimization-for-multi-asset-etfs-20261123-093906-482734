"""Price panel construction, arithmetic returns, and the risk-free rate.

Implements paper Section 2: daily arithmetic returns r_{i,t} = P_{i,t}/P_{i,t-1} - 1,
the common panel built by dropping any date with a missing observation (no
interpolation or forward filling of prices), and the risk-free return
r_{f,t} = (1 + y_t/100)^(1/252) - 1 from the most recently available
three-month Treasury par yield quotation on or before each date.
"""

import numpy as np
import pandas as pd

from data.data_loader import load_ohlcv, load_risk_free_yield


def build_price_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    identity_date: str | None = None,
) -> pd.DataFrame:
    """Wide adjusted-close panel (date index, one column per ticker).

    Market data: QuantX Dataloader — https://quantx.limex.pro/
    Bars are loaded dividend-adjusted (adjust_dividends=True, the Dataloader
    default), matching the paper's Yahoo Finance auto_adjust=True convention
    (Section 2, data requirements).
    """
    bars, meta = load_ohlcv(tickers, start_date, end_date, identity_date=identity_date)
    wide = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    wide = wide[tickers]
    before = len(wide)
    wide = wide.dropna(how="any")  # common panel via dropna(), no interpolation/ffill
    assert wide.shape[1] == len(tickers)
    return wide, meta, before


def arithmetic_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """r_{i,t} = P_{i,t}/P_{i,t-1} - 1 (paper Section 2)."""
    returns = price_panel / price_panel.shift(1) - 1.0
    return returns.dropna(how="any")


def build_risk_free_returns(dates: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    """Daily risk-free return aligned to ``dates`` using the most recently
    available Treasury quotation on or before each date (paper Section 2)."""
    yield_series = load_risk_free_yield(start_date, end_date)
    yield_series = yield_series.reindex(
        pd.date_range(yield_series.index.min(), max(yield_series.index.max(), dates.max()), freq="D")
    ).ffill()
    aligned_yield = yield_series.reindex(dates).ffill()
    assert aligned_yield.isna().sum() == 0, "risk-free yield could not be aligned to every return date"
    daily_rf = (1.0 + aligned_yield / 100.0) ** (1.0 / 252.0) - 1.0
    daily_rf.name = "rf"
    return daily_rf


def verify_calendar(dates: pd.DatetimeIndex) -> dict:
    """Basic sanity checks: sorted, no duplicates (paper Section 2, "verify no
    missing/extra/duplicated sessions"). An exact NYSE calendar feed is not
    available in this environment; duplicate/order checks are performed
    directly on the loaded panel instead (paper does not name its exact
    calendar source)."""
    assert dates.is_monotonic_increasing, "return dates are not sorted"
    assert dates.is_unique, "duplicated session dates in the return panel"
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    return {"n_dates": len(dates), "max_gap_days": int(gaps.max()) if len(gaps) else 0}
