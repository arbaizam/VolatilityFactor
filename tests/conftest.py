"""Shared pytest fixtures: the fixture price extract, test config and golden outputs.

Every test runs with the fixture data and config as the active defaults.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volatility_factor import Config, PricingData, set_config, set_data

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "golden"


@pytest.fixture(scope="session")
def pricing() -> PricingData:
    """The fixture price extract (tests/fixtures/pricing_sample.csv)."""
    return PricingData.from_csv(FIXTURES / "pricing_sample.csv")


@pytest.fixture(scope="session")
def config() -> Config:
    """The test config (tests/fixtures/config_test.yml)."""
    return Config.load(FIXTURES / "config_test.yml")


@pytest.fixture(autouse=True)
def _defaults(pricing, config):
    """Every test runs against the fixture data/config unless it says otherwise."""
    set_config(config)
    set_data(pricing)
    yield
    set_config(None)
    set_data(None)


@pytest.fixture(scope="session")
def golden():
    """Loader for the gzipped golden outputs, cached per file."""
    cache: dict[str, object] = {}

    def load(name: str):
        """Load one golden file by name (e.g. ``"pipeline"``)."""
        if name not in cache:
            cache[name] = pd.read_pickle(GOLDEN / f"{name}.pkl.gz")
        return cache[name]

    return load
