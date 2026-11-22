"""Volatility factor VaR models: GARCH simulation, circular block bootstrap,
backtesting and performance monitoring."""

from volatility_factor.backtesting import (
    BacktestRun,
    VarBacktest,
    log_likelihood_test,
    run_backtests,
)
from volatility_factor.config import (
    Config,
    FactorConfig,
    GarchSettings,
    get_config,
    set_config,
    use_config,
)
from volatility_factor.data import PricingData, VarData, get_data, set_data, use_data
from volatility_factor.garch import GarchFit, GarchSim
from volatility_factor.monitoring import MonitoringPeriods
from volatility_factor.nonparametric import NonParametricVar
from volatility_factor.periods import (
    create_periods,
    get_stress_range,
    get_stress_range_multi,
    get_stress_range_single,
)
from volatility_factor.pipeline import (
    BootstrapRun,
    GarchRun,
    run_circular_bootstrap,
    run_garch_var,
)
from volatility_factor.results import VaRResult, to_long

__version__ = "1.0.0"

__all__ = [
    "BacktestRun",
    "BootstrapRun",
    "Config",
    "FactorConfig",
    "GarchFit",
    "GarchRun",
    "GarchSettings",
    "GarchSim",
    "MonitoringPeriods",
    "NonParametricVar",
    "PricingData",
    "VaRResult",
    "VarBacktest",
    "VarData",
    "create_periods",
    "get_config",
    "get_data",
    "get_stress_range",
    "get_stress_range_multi",
    "get_stress_range_single",
    "log_likelihood_test",
    "run_backtests",
    "run_circular_bootstrap",
    "run_garch_var",
    "set_config",
    "set_data",
    "to_long",
    "use_config",
    "use_data",
]
