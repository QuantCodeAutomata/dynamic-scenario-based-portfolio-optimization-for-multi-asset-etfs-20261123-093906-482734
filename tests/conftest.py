"""Shared pytest fixtures. Market-data tests use a small, cached slice of the
real ETF panel loaded once per session through data/data_loader.py."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced real-data slice via the QuantX Dataloader"
    )


@pytest.fixture(scope="session")
def market_data_available() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def small_price_panel(request, market_data_available):
    """A few instruments of the paper's universe over roughly two years,
    cached across test runs so repeated runs do not re-download it."""
    if not market_data_available:
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    cache = request.config.cache
    cached = cache.get("market_data/small_panel", None)
    import pandas as pd

    from src.returns import build_price_panel

    if cached is not None:
        return pd.DataFrame(cached["data"]).assign(
            **{"__date__": pd.to_datetime(cached["index"])}
        ).set_index("__date__").rename_axis("date")

    tickers = ["VT", "VTI", "ACWI", "BND", "QAI"]
    panel, _meta, _before = build_price_panel(tickers, "2023-01-01", "2025-01-01")
    cache.set(
        "market_data/small_panel",
        {"data": panel.reset_index(drop=True).to_dict(orient="list"), "index": panel.index.astype(str).tolist()},
    )
    return panel
