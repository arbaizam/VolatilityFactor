"""VaR method implementations.

Import concrete classes from their modules, for example:
``from volatility_factor.methods.historical import HistoricalSimulationVaR``.
This avoids importing heavier numerical dependencies until the selected method
is used.
"""

__all__ = [
    "CircularBlockBootstrapVaR",
    "CornishFisherVaR",
    "EVTPeaksOverThresholdVaR",
    "EWMAVaR",
    "FilteredHistoricalSimulationVaR",
    "GarchSimulation",
    "HistoricalSimulationVaR",
    "ParametricVaR",
]
