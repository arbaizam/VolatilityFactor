"""GARCH(1,1) estimation and custom n-day price simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from volatility_factor._kernels import nday_price_sim
from volatility_factor._utils import validate_date
from volatility_factor.data import VarData
from volatility_factor.results import VaRResult

if TYPE_CHECKING:
    import plotly.graph_objects as go

SHOCK_TYPES = ("normal", "custom_normal", "bootstrap", "custom_bootstrap")
Y_SCALES = (1000, 10000, 100, 10)  # tried in this order until the optimiser converges


@dataclass(frozen=True)
class GarchFit:
    """One fitted GARCH model: the full ``arch`` result plus how it was obtained.

    Every attribute of :class:`arch.univariate.base.ARCHModelResult` is available
    directly (``fit.aic``, ``fit.params``, ``fit.summary()``, ``fit.forecast(...)``,
    ``fit.std_resid`` ...), or through ``fit.result``.

    Attributes:
        ticker: the constituent this model was fitted to.
        result: the ``arch`` result object.
        yscale: factor the log returns were multiplied by before fitting.
        attempts: ``(yscale, convergence_flag)`` for every scale tried.
    """

    ticker: str
    result: ARCHModelResult
    yscale: int
    attempts: tuple[tuple[int, int], ...]

    def __getattr__(self, name: str) -> Any:
        """Look up any other attribute on the ``arch`` result."""
        if name in {"result", "__setstate__", "__getstate__"}:
            raise AttributeError(name)
        return getattr(self.result, name)

    def __dir__(self) -> list[str]:
        """Own attributes plus those of the ``arch`` result (for tab completion)."""
        return sorted(set(super().__dir__()) | set(dir(self.result)))

    @property
    def convergence_code(self) -> int:
        """Optimiser convergence flag of the final fit (0 = converged)."""
        return int(self.result.convergence_flag)

    @property
    def coefficients(self) -> list[float]:
        """``[omega, alpha[1], beta[1]]`` on the scaled returns."""
        p = self.result.params
        return [p["omega"], p["alpha[1]"], p["beta[1]"]]

    @property
    def constant_mean(self) -> float:
        """``mu`` for a constant-mean model, else 0 (on the scaled returns)."""
        return self.result.params["mu"] if self.result.num_params == 4 else 0

    def __repr__(self) -> str:
        """Ticker, scale, convergence code and parameters."""
        return (
            f"GarchFit(ticker={self.ticker!r}, yscale={self.yscale}, "
            f"convergence_code={self.convergence_code}, params={dict(self.result.params)})"
        )


class GarchSim(VarData):
    """GARCH(1,1) models and simulation for one index or a weighted portfolio.

    One GARCH(1,1) (normal errors) is fitted per constituent on the sample-window log
    returns. Fitting is lazy: it runs on first access to a fitted value, or when
    :meth:`fit` is called.

    Args:
        tickers: index ticker(s).
        weights: portfolio weights (``None`` for equal weights).
        sample_start: first date of the training window (``YYYY-MM-DD``).
        sample_end: last date of the training window (``YYYY-MM-DD``).
        mean_model: ``arch`` mean model. Defaults to ``"Zero"``.
        vol_model: ``arch`` volatility model. Defaults to ``"Garch"``.
        random_seed: seed for shock generation. Defaults to 1.
        data, config, volatility_factor_id: see :class:`~volatility_factor.data.VarData`.

    Example::

        res = (GarchSim.from_id("agy_cmo", sample_start="2016-01-01", sample_end="2024-12-31")
                 .fit()
                 .simulate(horizon1=63, horizon2=261, ndays=10, nsims=10_000,
                           shock_type="custom_normal", start_vals=("2024-01-01", 0.5)))
        res["VaR_99"]
    """

    _FIT_CACHE = (
        "fits",
        "training_data",
        "training_data_scaled",
        "residuals",
        "std_residuals",
        "conditional_vol",
        "std_returns",
    )

    def __init__(
        self,
        tickers: str | Sequence[str],
        weights: Sequence[float] | float | None,
        sample_start: str,
        sample_end: str,
        mean_model: str = "Zero",
        vol_model: str = "Garch",
        *,
        random_seed: int = 1,
        **kwargs: Any,
    ):
        """Set up the model; fitting happens on first use. See the class docstring."""
        super().__init__(tickers, weights, **kwargs)
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.random_seed = random_seed
        self.sim_paths: np.ndarray | None = None

    # --------------------------------------------------------------- settings
    def _reset_fit(self) -> None:
        """Drop the fitted models and everything derived from them."""
        for name in self._FIT_CACHE:
            self.__dict__.pop(name, None)

    @property
    def sample_start(self) -> str:
        """First date of the training window (``YYYY-MM-DD``)."""
        return self._sample_start

    @sample_start.setter
    def sample_start(self, value: str) -> None:
        """Validate and set the window start, clearing any fit."""
        self._sample_start = validate_date(value, "sample_start")
        self._reset_fit()

    @property
    def sample_end(self) -> str:
        """Last date of the training window (``YYYY-MM-DD``)."""
        return self._sample_end

    @sample_end.setter
    def sample_end(self, value: str) -> None:
        """Validate and set the window end, clearing any fit."""
        self._sample_end = validate_date(value, "sample_end")
        self._reset_fit()

    def with_sample(self, sample_start: str, sample_end: str) -> GarchSim:
        """Change the training window (clears any fit) and return ``self``."""
        self.sample_start = sample_start
        self.sample_end = sample_end
        return self

    # ------------------------------------------------------------------ fitting
    def fit(self) -> GarchSim:
        """Fit the models now (otherwise they are fitted on first use). Returns ``self``."""
        _ = self.fits
        return self

    @cached_property
    def training_data(self) -> pd.DataFrame:
        """Constituent one-day log returns in the sample window."""
        return self.log_return_matrix[self.sample_start : self.sample_end].copy()

    @cached_property
    def fits(self) -> dict[str, GarchFit]:
        """Fitted models keyed by ticker."""
        fits = {}
        for col in self.training_data.columns:
            attempts = []
            for y_scale in Y_SCALES:
                scaled = self.training_data[col] * y_scale
                result = arch_model(
                    y=scaled,
                    mean=self.mean_model,
                    vol=self.vol_model,
                    p=1,
                    o=0,
                    q=1,
                    dist="Normal",
                    rescale=False,
                ).fit(disp="off", show_warning=False)
                flag = int(result.convergence_flag)
                attempts.append((y_scale, flag))
                if flag == 0:
                    break
            fits[col] = GarchFit(ticker=col, result=result, yscale=y_scale, attempts=tuple(attempts))
        return fits

    def _fit_frame(self, getter) -> pd.DataFrame:
        """One column per ticker built from each fit, on the training-data index.

        Args:
            getter: function from a :class:`GarchFit` to an array of training-length values.
        """
        return pd.DataFrame(
            {col: getter(fit) for col, fit in self.fits.items()},
            index=self.training_data.index,
            columns=self.training_data.columns,
        )

    @cached_property
    def training_data_scaled(self) -> pd.DataFrame:
        """Training data multiplied by each constituent's ``yscale``."""
        return self._fit_frame(lambda f: (self.training_data[f.ticker] * f.yscale).values)

    @cached_property
    def residuals(self) -> pd.DataFrame:
        """Fitted residuals (on the scaled returns), one column per ticker."""
        return self._fit_frame(lambda f: f.result.resid.values)

    @cached_property
    def std_residuals(self) -> pd.DataFrame:
        """Standardized residuals, one column per ticker."""
        return self._fit_frame(lambda f: f.result.std_resid.values)

    @cached_property
    def conditional_vol(self) -> pd.DataFrame:
        """Fitted conditional volatility (on the scaled returns), one column per ticker."""
        return self._fit_frame(lambda f: f.result.conditional_volatility.values)

    @cached_property
    def std_returns(self) -> pd.DataFrame:
        """Scaled training data divided by conditional volatility."""
        return self.training_data_scaled / self.conditional_vol

    @property
    def garch_fit(self) -> list[ARCHModelResult]:
        """``arch`` result objects, one per ticker (in ticker order)."""
        return [f.result for f in self.fits.values()]

    @property
    def yscale(self) -> list[int]:
        """Scale applied to each ticker's returns before fitting."""
        return [f.yscale for f in self.fits.values()]

    @property
    def convergence_codes(self) -> list[int]:
        """Convergence flag of each ticker's final fit (0 = converged)."""
        return [f.convergence_code for f in self.fits.values()]

    @property
    def coefficients(self) -> list[list[float]]:
        """``[omega, alpha, beta]`` for each ticker."""
        return [f.coefficients for f in self.fits.values()]

    @property
    def constant_mean(self) -> list[float]:
        """Constant mean for each ticker (0 for the zero-mean model)."""
        return [f.constant_mean for f in self.fits.values()]

    def summary(self) -> pd.DataFrame:
        """One row per constituent: parameters, scale and fit statistics."""
        rows = []
        for ticker, f in self.fits.items():
            rows.append({
                "Ticker": ticker,
                "yscale": f.yscale,
                "ConvergenceCode": f.convergence_code,
                **{name: value for name, value in f.result.params.items()},
                "LogLikelihood": f.result.loglikelihood,
                "AIC": f.result.aic,
                "BIC": f.result.bic,
                "NObs": f.result.nobs,
            })
        return pd.DataFrame(rows).set_index("Ticker")

    # ------------------------------------------------------------------- shocks
    def get_normal_shocks(self, horizon: int, nsims: int) -> np.ndarray:
        """Standard normal shocks; correlated across constituents for a portfolio."""
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index is True:
            shocks = random_state.standard_normal(size=(nsims, horizon))
        else:
            sigma = self.std_returns.corr().values
            shocks = random_state.multivariate_normal(
                mean=np.zeros(len(self.tickers)), cov=sigma, size=(nsims, horizon)
            )
        return shocks

    def get_custom_normal_shocks(self, horizon1: int, horizon2: int, nsims: int) -> np.ndarray:
        """Normal shocks with the first five forecast days scaled by sqrt(2)."""
        horizon = horizon1 + horizon2
        shocks = self.get_normal_shocks(horizon, nsims)
        shocks[:, horizon1 : (horizon1 + 5)] *= np.sqrt(2)
        return shocks

    def get_bootstrap_shocks(
        self, horizon: int, nsims: int, bootstrap_start: str, bootstrap_end: str
    ) -> np.ndarray:
        """Shocks resampled from fitted standardized residuals in the bootstrap window."""
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index is True:
            fitted_shocks = self.std_residuals
            boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
            shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon))
            shocks = boot_shocks[shock_idx]
        else:
            fitted_shocks = self.std_returns
            boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
            shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon))
            shocks = np.empty(shape=(nsims, horizon, len(self.tickers)))
            for i in range(nsims):
                shocks[i, :, :] = boot_shocks[shock_idx[i]]
        return shocks

    def get_custom_boot_shocks(
        self,
        horizon1: int,
        horizon2: int,
        nsims: int,
        bootstrap_start: str,
        bootstrap_end: str,
    ) -> np.ndarray:
        """Bootstrapped shocks for ``horizon1``, then normal shocks for ``horizon2``."""
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index is True:
            fitted_shocks = self.std_residuals
            h2shocks = random_state.standard_normal(size=(nsims, horizon2))
        else:
            fitted_shocks = self.std_returns
            sigma = fitted_shocks.corr().values
            h2shocks = random_state.multivariate_normal(
                mean=np.zeros(fitted_shocks.shape[1]), cov=sigma, size=(nsims, horizon2)
            )
        boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
        shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon1))
        h1shocks = boot_shocks[shock_idx]
        return np.hstack((h1shocks, h2shocks))

    # ---------------------------------------------------------- starting values
    nday_price_sim = staticmethod(nday_price_sim)

    def get_single_index_startvals(self, start_vals: Sequence) -> tuple[float, float]:
        """Starting residual and sigma at the given quantile of one-step forecast variance.

        Args:
            start_vals: ``(date, quantile)``; forecasts from ``date`` onward are ranked.
        """
        start_date, quantile = start_vals
        fc = self.garch_fit[0].forecast(horizon=1, start=start_date, reindex=False).variance
        dfstart = pd.concat([self.residuals, self.conditional_vol], axis=1).loc[start_date:]
        idx = fc[fc.iloc[:, 0] == fc.iloc[:, 0].quantile(quantile, interpolation="higher")].index
        idx = pd.to_datetime(idx).strftime("%Y-%m-%d")[0]
        start = dfstart[idx:idx].values.squeeze()
        return start[0], start[1]

    def get_multi_index_startvals(self, start_vals: Sequence) -> tuple[np.ndarray, np.ndarray]:
        """Portfolio version of :meth:`get_single_index_startvals`.

        Dates are ranked by portfolio volatility built from one-step constituent forecasts
        and the standardized-residual correlation matrix.
        """
        start_date, quantile = start_vals
        sigma_step1 = np.empty_like(self.training_data_scaled[start_date:])
        for i in range(len(self.tickers)):
            sigma_step1[:, i] = np.sqrt(
                self.garch_fit[i]
                .forecast(horizon=1, start=start_date, reindex=False)
                .variance.values.squeeze()
            )
        rho = self.std_residuals.corr().values
        portfolio_sigma = []
        for i in range(len(sigma_step1)):
            m_sigma = np.diag(sigma_step1[i])
            sigma2 = m_sigma.dot(rho).dot(m_sigma)
            sigma2_weight = self.weights.dot(sigma2).dot(self.weights)
            portfolio_sigma.append(np.sqrt(sigma2_weight))
        sigma_p = pd.Series(data=portfolio_sigma, index=self.training_data[start_date:].index)
        idx = sigma_p[sigma_p == sigma_p.quantile(quantile, interpolation="higher")].index
        idx = pd.to_datetime(idx).strftime("%Y-%m-%d")[0]
        dfstart = pd.concat([self.residuals, self.conditional_vol], axis=1).loc[start_date:,]
        start = dfstart[idx:idx].values.squeeze()
        start = np.transpose(start.reshape(2, len(self.tickers)))
        return start[:, 0], start[:, 1]

    # --------------------------------------------------------------- simulation
    def _shocks(
        self,
        shock_type: str,
        horizon1: int,
        horizon2: int,
        nsims: int,
        bootstrap_start: str | None,
        bootstrap_end: str | None,
    ) -> np.ndarray:
        """Generate the shock array for ``shock_type``.

        Raises:
            ValueError: for an unknown shock type, or a bootstrap type without its window.
        """
        horizon = horizon1 + horizon2
        if shock_type in ("bootstrap", "custom_bootstrap") and (
            bootstrap_start is None or bootstrap_end is None
        ):
            raise ValueError(f"bootstrap_start and bootstrap_end are required for {shock_type!r}")
        if shock_type == "normal":
            return self.get_normal_shocks(horizon, nsims)
        if shock_type == "custom_normal":
            return self.get_custom_normal_shocks(horizon1, horizon2, nsims)
        if shock_type == "bootstrap":
            return self.get_bootstrap_shocks(horizon, nsims, bootstrap_start, bootstrap_end)
        if shock_type == "custom_bootstrap":
            return self.get_custom_boot_shocks(
                horizon1, horizon2, nsims, bootstrap_start, bootstrap_end
            )
        raise ValueError(f"shock_type must be one of the following: {', '.join(SHOCK_TYPES)}")

    def simulate(
        self,
        horizon1: int,
        horizon2: int,
        ndays: int,
        shock_type: str,
        nsims: int,
        var_p: float = 0.99,
        start_vals: Sequence | None = None,
        bootstrap_start: str | None = None,
        bootstrap_end: str | None = None,
    ) -> VaRResult:
        """Simulate price paths and compute n-day VaR.

        Args:
            horizon1: lag; the first ``horizon1`` simulated days are dropped.
            horizon2: days simulated after the lag (usually ``252 + ndays - 1``).
            ndays: liquidation period in days.
            shock_type: ``"normal"``, ``"custom_normal"``, ``"bootstrap"`` or
                ``"custom_bootstrap"``.
            nsims: number of simulated paths.
            var_p: confidence level used for each path's VaR. Defaults to 0.99.
            start_vals: ``(date, quantile)`` for the starting residual/sigma; ``None``
                uses the last values in the sample.
            bootstrap_start, bootstrap_end: bootstrap window (required for the two
                bootstrap shock types).

        Returns:
            A :class:`~volatility_factor.results.VaRResult`.
        """
        horizon = horizon1 + horizon2
        shocks = self._shocks(shock_type, horizon1, horizon2, nsims, bootstrap_start, bootstrap_end)

        if self.single_index:
            coefs = np.array(self.coefficients[0]).squeeze()
            mean = np.array(self.constant_mean[0])
            if start_vals is None:
                resid_start = self.residuals.iloc[-1, 0]
                sigma_start = self.conditional_vol.iloc[-1, 0]
            else:
                resid_start, sigma_start = self.get_single_index_startvals(start_vals)
            ysims, _, _ = nday_price_sim(horizon=horizon, shocks=shocks, coefs=coefs, mean=mean,
                                         resid_start=resid_start, sigma_start=sigma_start,
                                         nsims=nsims)
            sim_paths = ysims / self.yscale[0]
        else:
            if start_vals is not None:
                resid_starts, sigma_starts = self.get_multi_index_startvals(start_vals)
            ysims = np.empty_like(shocks)
            residsims = np.empty_like(shocks)
            sigmasims = np.empty_like(shocks)
            for i in range(len(self.tickers)):
                coefs = np.array(self.coefficients[i]).squeeze()
                mean = np.array(self.constant_mean[i])
                if start_vals is None:
                    resid_start = self.residuals.iloc[-1, i]
                    sigma_start = self.conditional_vol.iloc[-1, i]
                else:
                    resid_start = resid_starts[i]
                    sigma_start = sigma_starts[i]
                ysims[:, :, i], residsims[:, :, i], sigmasims[:, :, i] = nday_price_sim(
                    horizon=horizon, shocks=shocks[:, :, i], coefs=coefs, mean=mean,
                    resid_start=resid_start, sigma_start=sigma_start, nsims=nsims,
                )
                ysims[:, :, i] = ysims[:, :, i] / self.yscale[i]
            sim_paths = np.empty(shape=(nsims, horizon))
            for i in range(nsims):
                sim_paths[i, :] = ysims[i, :, :].dot(self.weights)
        self.sim_paths = sim_paths

        sim_paths_sub = sim_paths[:, horizon1:]
        ndaypc = pd.DataFrame(sim_paths_sub.T).rolling(ndays).sum().dropna().reset_index(drop=True)
        var = ndaypc.quantile(1 - var_p, interpolation="higher") * 100  # log percent

        record = self._result_record(
            var, horizon1, horizon2, ndays, shock_type, nsims, var_p, start_vals,
            bootstrap_start, bootstrap_end,
        )
        return VaRResult(record=record, sim_paths=sim_paths, nday_returns=ndaypc, var_by_path=var)

    def _result_record(
        self,
        var: pd.Series,
        horizon1: int,
        horizon2: int,
        ndays: int,
        shock_type: str,
        nsims: int,
        var_p: float,
        start_vals: Sequence | None,
        bootstrap_start: str | None,
        bootstrap_end: str | None,
    ) -> dict[str, Any]:
        """Build the result dict for one simulation.

        Args:
            var: per-path VaR quantiles (log percent).
            horizon1, horizon2, ndays, shock_type, nsims, var_p, start_vals,
                bootstrap_start, bootstrap_end: the simulation settings, recorded as-is.

        Returns:
            Dict of settings, convergence codes, and log and arithmetic VaR at the 50th,
            75th, 95th and 99th percentiles across paths.
        """
        is_boot = shock_type in ["bootstrap", "custom_bootstrap"]

        def q(level: float) -> float:
            """Quantile of the per-path VaR at ``level`` (the left tail)."""
            return var.quantile(1 - level, interpolation="higher")

        return {
            "Tickers": self.ticker_description,
            "Weights": self.weight_description,
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "BootstrapStart": bootstrap_start if is_boot else None,
            "BootstrapEnd": bootstrap_end if is_boot else None,
            "ShockType": shock_type,
            "VaRp": str(round(var_p * 100)) + "Q",
            "MeanModel": self.mean_model,
            "VolatilityModel": self.vol_model,
            "Method": (
                f"GarchSim_{shock_type}_{str(round(start_vals[1] * 100))}"
                if start_vals is not None
                else f"GarchSim_{shock_type}"
            ),
            "Horizon1": horizon1,
            "Horizon2": horizon2,
            "NDays": ndays,
            "Simulations": nsims,
            "StartingValueQ": (
                str(round(start_vals[1] * 100)) + "Q" if start_vals is not None else "None"
            ),
            "StartingValueDate": start_vals[0] if start_vals is not None else "None",
            "ConvergencyCode": ";".join([str(x) for x in self.convergence_codes]),
            "LogVaR_50": round(q(0.50) * -1, 4),
            "LogVaR_75": round(q(0.75) * -1, 4),
            "LogVaR_95": round(q(0.95) * -1, 4),
            "LogVaR_99": round(q(0.99) * -1, 4),
            "VaR_50": round(-(np.exp(q(0.5) / 100) - 1) * 100, 4),
            "VaR_75": round(-(np.exp(q(0.75) / 100) - 1) * 100, 4),
            "VaR_95": round(-(np.exp(q(0.95) / 100) - 1) * 100, 4),
            "VaR_99": round(-(np.exp(q(0.99) / 100) - 1) * 100, 4),
        }

    # ------------------------------------------------------------------ plotting
    def plot_fit(self, ticker: str | None = None, **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.garch_fit_plot`."""
        from volatility_factor.plotting import garch_fit_plot

        return garch_fit_plot(self, ticker=ticker, **kwargs)
