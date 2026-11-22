# volatility-factor

GARCH(1,1) simulation and circular block bootstrap VaR for volatility factors, with
backtesting (Kupiec / Christoffersen) and quarterly performance monitoring. Charts use Plotly.

## Install

```bash
pip install -e ".[dev]"      # from a clone, for development
pip install volatility-factor  # once published
```

Python 3.10+. Numerical results are regression-tested in the reference environment
(numpy 2.2, pandas 2.2, scipy 1.15, arch 8.0, numba 0.61).

## Quick start

```python
import volatility_factor as vf

# A configured factor, by id or by tickers
res = (
    vf.VarData.from_id("agy_cmo")
      .garch(sample_start="2016-01-01", sample_end="2024-12-31")
      .simulate(horizon1=63, horizon2=261, ndays=10, nsims=10_000,
                shock_type="custom_normal", start_vals=("2024-01-01", 0.5))
)
res["VaR_99"]           # result behaves like the original dict
res.plot_distribution() # Plotly figure

# Everything in the config, saved to the output folder
run = vf.run_garch_var().save()
run.backtest().save()
vf.run_circular_bootstrap().save()
```

## Configuration and data

Nothing is read at import time.

| What | Lookup order |
|---|---|
| Config | `config=` argument → `vf.set_config()` / `vf.use_config()` → `$VOLFACTOR_CONFIG` → `configs/var_config_volfactor.yml` in the working directory or a parent |
| Pricing data | `data=` argument → `vf.set_data()` / `vf.use_data()` → `$VOLFACTOR_DATA` → `data_settings.pricing_data` in the config |

Pricing data can be a CSV or parquet path, a DataFrame, a `PricingData`, or a Delta table
(`PricingData.from_delta("catalog.schema.table")`). The format is long:
`Date, Ticker, Source, Description, Price`. Files loaded by path are cached and shared.

```python
with vf.use_data("tests/fixtures/pricing_sample.csv"):
    vf.VarData("CMOS").plot_rolling_volatility()

cfg = vf.Config.load("configs/var_config_volfactor.yml").with_garch(nsims=2_000).with_factors("agy_cmo")
vf.run_garch_var(cfg, data="other_prices.parquet")
```

Relative paths in the config file (`pricing_data`, `loc_output`) are relative to the config
file itself.

## Package layout

```
src/volatility_factor/
  config.py         Config, FactorConfig, GarchSettings; set_config / use_config
  data.py           PricingData, VarData; set_data / use_data
  garch.py          GarchSim, GarchFit (wraps the full arch result)
  nonparametric.py  NonParametricVar (circular block bootstrap)
  backtesting.py    VarBacktest, log_likelihood_test, run_backtests
  monitoring.py     MonitoringPeriods, F-tests, exception counts, shortfalls
  periods.py        create_periods, stress-window selection
  pipeline.py       run_garch_var, run_circular_bootstrap (config-driven runs)
  plotting.py       Plotly charts (every function returns a go.Figure)
  results.py        VaRResult, to_long
notebooks/          main run, backtesting, monitoring, project overview
configs/            var_config_volfactor.yml
tests/              unit tests + exact regression tests against the pre-refactor code
```

### GARCH fits

Each constituent's fit is a `GarchFit` holding the full `arch` result, so nothing has to
be copied out by hand:

```python
g = vf.GarchSim.from_id("agy_cmo", sample_start="2016-01-01", sample_end="2024-12-31").fit()
fit = g.fits["CMOS"]
fit.aic, fit.params, fit.summary(), fit.forecast(horizon=5)   # any arch attribute
fit.yscale, fit.attempts                                       # how it was fitted
g.summary()                                                    # one row per constituent
g.residuals, g.conditional_vol, g.std_residuals                # DataFrames, as before
```

## Tests

```bash
pytest -m "not regression"   # fast unit tests
pytest -m regression         # exact comparison with outputs captured from the original code
```

The regression goldens (`tests/fixtures/golden`) were captured from commit `d0d285f` with
`tests/regression/capture_legacy_goldens.py`, using `tests/fixtures/pricing_sample.csv`
(a subset of the full price file; results on it are identical to the full file).

## Renamed API

The original CamelCase methods and keyword arguments still work and emit a
`DeprecationWarning`:

| Before | Now |
|---|---|
| `GarchSim.CustomNDayPriceSimulation(..., VaRp=)` | `simulate(..., var_p=)` → `VaRResult`, or `custom_nday_price_simulation()` → dict |
| `NonParametricVar.CircularBlockBootstrap(..., VaRp=)` | `circular_block_bootstrap(..., var_p=)` |
| `VarBacktest(VaR=, VaR_is_log=)` | `VarBacktest(var=, var_is_log=)` |
| `LogLikelihood_test` (notebook) | `log_likelihood_test` / `VarBacktest.independence_test()` |
| `util_functions.*` | `volatility_factor.periods.*` |
| `constants.CONFIGS` | `vf.get_config()` |
