# volatility-factor

GARCH(1,1) simulation and circular block bootstrap VaR for volatility factors, with
backtesting (Kupiec / Christoffersen) and quarterly performance monitoring. Charts use Plotly.

## Install

```bash
pip install -e ".[dev]"      # from a clone, for development
pip install volatility-factor  # once published
```

Use Python 3.10–3.13. The package installs the numerical dependencies at the versions used
for the reference VaR results.

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
res["VaR_99"]           # read a VaR result by column name
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

Pricing data can be a CSV or parquet path, a DataFrame, or a `PricingData` instance.
Use long-format rows with `Date, Ticker, Source, Description, Price`.
Files loaded by path are cached and shared.

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
  data.py           PricingData (incl. price-observation QA), VarData; set_data / use_data
  garch.py          GarchSim, GarchFit (wraps the full arch result)
  nonparametric.py  NonParametricVar (circular block bootstrap)
  backtesting.py    VarBacktest, log_likelihood_test, run_backtests
  monitoring.py     MonitoringPeriods, F-tests, exception counts, shortfalls
  periods.py        create_periods, stress-window selection
  pipeline.py       run_garch_var, run_circular_bootstrap (config-driven runs)
  plotting.py       Plotly charts (every function returns a go.Figure)
  results.py        VaRResult, to_long
notebooks/          main run, backtesting, monitoring, ProjectOverview (end-to-end guide)
configs/            var_config_volfactor.yml
docs/               ProjectOverview.html (executed export of the overview notebook)
tests/              unit tests + exact regression tests against reference outputs
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
g.residuals, g.conditional_vol, g.std_residuals                # DataFrames
```

### Price-observation QA

`coverage_report()` flags missing vendor dates, long gaps, stale prices, duplicate or
invalid dates, missing vendor sources, missing or invalid prices, and indices that stop
early. Vendor pricing activity determines the expected dates.

```python
data = vf.get_data()
data.coverage_report(start="2016-01-01").query("Status == 'Review'")
data.gaps("G0BA")                         # each gap for one index
data.plot_observation_counts(freq="YE")   # coverage heatmap by index and year
vf.VarData.from_id("G6P0[50];G9P0[50]").coverage_report()  # + dates only one constituent has
```

Portfolio coverage also flags a lack of consecutive aligned prices. The GARCH batch
writes its sample-coverage workbook if every model is skipped and raises an error before
publishing new VaR results.

### Backtesting

Batch backtesting assigns each VaR quantile its expected exception rate. For example,
95% VaR uses a 5% rate. The test report includes the number of usable returns. A window
with zero returns has blank statistics and a blank independence decision.

## Tests

```bash
pytest -m "not regression"   # fast unit tests
pytest -m regression         # checks against reference outputs
ruff check src tests         # lint and docstring checks
```

The regression suite uses the reference outputs in `tests/fixtures/golden` and the
price extract in `tests/fixtures/pricing_sample.csv`. Reported VaR must match exactly.
Unrounded calculations use tight tolerances for differences between numerical libraries.

To re-run and re-export the overview notebook (needs the `dev` extras):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ProjectOverview.ipynb
jupyter nbconvert --to html notebooks/ProjectOverview.ipynb --output-dir docs
```
