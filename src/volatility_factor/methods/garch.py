"""GARCH simulation VaR model.

This module ports the existing ``GarchSim`` behavior while making pricing data
an explicit constructor argument. The legacy method name is retained as an
alias for regression testing against the current notebook output.
"""

from __future__ import annotations

import re
from typing import Sequence

from arch import arch_model
from numba import jit
import numpy as np
import pandas as pd

from volatility_factor.portfolio import PortfolioPricingData


class GarchSimulation(PortfolioPricingData):
    """GARCH(1,1) estimation and custom n-day price simulation."""

    def __init__(
        self,
        tickers: str | Sequence[str],
        weights: Sequence[float] | None,
        pricing_data: pd.DataFrame,
        sample_start: str,
        sample_end: str,
        mean_model: str = "Zero",
        vol_model: str = "Garch",
        volatility_factor_id: str | None = None,
        random_seed: int = 1,
    ):
        super().__init__(
            tickers=tickers,
            weights=weights,
            pricing_data=pricing_data,
            volatility_factor_id=volatility_factor_id,
        )
        self.sample_start = self._validate_date(sample_start, "sample_start")
        self.sample_end = self._validate_date(sample_end, "sample_end")
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.random_seed = random_seed
        self.sim_paths: np.ndarray | None = None
        self._reset_fit_cache()

    @staticmethod
    def _validate_date(value: str, field_name: str) -> str:
        if re.match(r"^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            return value
        raise ValueError(f"{field_name} must be formatted as string (YYYY-MM-DD)")

    def _reset_fit_cache(self) -> None:
        self._training_data = None
        self._training_data_scaled = None
        self._conditional_vol = None
        self._residuals = None
        self._std_residuals = None
        self._std_returns = None
        self._coefficients = None
        self._constant_mean = None
        self._garch_fit = None
        self._yscale = None
        self._convergence_codes = None

    @property
    def training_data(self) -> pd.DataFrame:
        if self._training_data is None:
            self._estimate_garch()
        return self._training_data

    @property
    def training_data_scaled(self) -> pd.DataFrame:
        if self._training_data_scaled is None:
            self._estimate_garch()
        return self._training_data_scaled

    @property
    def garch_fit(self) -> list:
        if self._garch_fit is None:
            self._estimate_garch()
        return self._garch_fit

    @property
    def yscale(self) -> list[int]:
        if self._yscale is None:
            self._estimate_garch()
        return self._yscale

    @property
    def convergence_codes(self) -> list[int]:
        if self._convergence_codes is None:
            self._estimate_garch()
        return self._convergence_codes

    @property
    def conditional_vol(self) -> pd.DataFrame:
        if self._conditional_vol is None:
            self._estimate_garch()
        return self._conditional_vol

    @property
    def residuals(self) -> pd.DataFrame:
        if self._residuals is None:
            self._estimate_garch()
        return self._residuals

    @property
    def std_residuals(self) -> pd.DataFrame:
        if self._std_residuals is None:
            self._estimate_garch()
        return self._std_residuals

    @property
    def coefficients(self) -> list[list[float]]:
        if self._coefficients is None:
            self._estimate_garch()
        return self._coefficients

    @property
    def constant_mean(self) -> list[float]:
        if self._constant_mean is None:
            self._estimate_garch()
        return self._constant_mean

    @property
    def std_returns(self) -> pd.DataFrame:
        if self._std_returns is None:
            self._estimate_garch()
        return self._std_returns

    def _estimate_garch(self) -> None:
        """Estimate one GARCH(1,1) model per constituent return series."""

        self._reset_fit_cache()
        training_data = self.log_return_matrix[self.sample_start : self.sample_end].copy()
        training_data_scaled = pd.DataFrame(
            data=None,
            columns=training_data.columns,
            index=training_data.index,
        )
        residuals = pd.DataFrame(data=None, columns=training_data.columns, index=training_data.index)
        std_residuals = pd.DataFrame(data=None, columns=training_data.columns, index=training_data.index)
        conditional_vol = pd.DataFrame(
            data=None,
            columns=training_data.columns,
            index=training_data.index,
        )
        coefficients: list[list[float]] = []
        constant_mean: list[float] = []
        y_scales: list[int] = []
        garch_fit: list = []
        convergence_codes: list[int] = []

        for column in training_data.columns:
            iterations = 0
            while iterations < 4:
                if iterations == 0:
                    y_scale = 1000
                elif iterations == 1:
                    y_scale = 10000
                elif iterations == 2:
                    y_scale = 100
                else:
                    y_scale = 10

                training_data_scaled_temp = training_data[column] * y_scale
                garch_model = arch_model(
                    y=training_data_scaled_temp,
                    mean=self.mean_model,
                    vol=self.vol_model,
                    p=1,
                    o=0,
                    q=1,
                    dist="Normal",
                    rescale=False,
                )
                fitted = garch_model.fit(disp="off", show_warning=False)
                convergence_flag = int(fitted.convergence_flag)
                if convergence_flag == 0:
                    break
                iterations += 1

            training_data_scaled[column] = training_data_scaled_temp
            residuals[column] = fitted.resid.values
            std_residuals[column] = fitted.std_resid.values
            conditional_vol[column] = fitted.conditional_volatility.values
            coefficients.append(
                [
                    fitted.params["omega"],
                    fitted.params["alpha[1]"],
                    fitted.params["beta[1]"],
                ]
            )
            constant_mean.append(fitted.params["mu"] if fitted.num_params == 4 else 0)
            garch_fit.append(fitted)
            y_scales.append(y_scale)
            convergence_codes.append(convergence_flag)

        self._training_data = training_data
        self._training_data_scaled = training_data_scaled
        self._yscale = y_scales
        self._garch_fit = garch_fit
        self._convergence_codes = convergence_codes
        self._residuals = residuals
        self._std_residuals = std_residuals
        self._coefficients = coefficients
        self._constant_mean = constant_mean
        self._std_returns = training_data_scaled / conditional_vol
        self._conditional_vol = conditional_vol

    def get_normal_shocks(self, horizon: int, nsims: int) -> np.ndarray:
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index:
            return random_state.standard_normal(size=(nsims, horizon))

        sigma = self.std_returns.corr().values
        return random_state.multivariate_normal(
            mean=np.zeros(len(self.tickers)),
            cov=sigma,
            size=(nsims, horizon),
        )

    def get_custom_normal_shocks(self, horizon1: int, horizon2: int, nsims: int) -> np.ndarray:
        horizon = horizon1 + horizon2
        shocks = self.get_normal_shocks(horizon, nsims)
        shocks[:, horizon1 : (horizon1 + 5)] *= np.sqrt(2)
        return shocks

    def get_bootstrap_shocks(
        self,
        horizon: int,
        nsims: int,
        bootstrap_start: str,
        bootstrap_end: str,
    ) -> np.ndarray:
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index:
            fitted_shocks = self.std_residuals
            boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
            shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon))
            return boot_shocks[shock_idx]

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
        random_state = np.random.RandomState(self.random_seed)
        if self.single_index:
            fitted_shocks = self.std_residuals
            h2shocks = random_state.standard_normal(size=(nsims, horizon2))
        else:
            fitted_shocks = self.std_returns
            sigma = fitted_shocks.corr().values
            h2shocks = random_state.multivariate_normal(
                mean=np.zeros(fitted_shocks.shape[1]),
                cov=sigma,
                size=(nsims, horizon2),
            )

        boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
        shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon1))
        h1shocks = boot_shocks[shock_idx]
        return np.hstack((h1shocks, h2shocks))

    @staticmethod
    @jit(nopython=True)
    def nday_price_sim(
        horizon: int,
        shocks: np.ndarray,
        coefs: np.ndarray,
        mean: np.ndarray,
        resid_start: float,
        sigma_start: float,
        nsims: int,
    ):
        resid_sim = np.empty(shape=(nsims, horizon + 1), dtype=np.float64)
        sigma_sim = np.empty(shape=(nsims, horizon + 1), dtype=np.float64)
        y_sim = np.empty(shape=(nsims, horizon), dtype=np.float64)

        for i in range(np.int64(nsims)):
            for j in range(horizon):
                resid_sim[i, 0] = resid_start
                sigma_sim[i, 0] = sigma_start
                yyy = np.array([1, resid_sim[i, j] ** 2, sigma_sim[i, j] ** 2])
                sigma_sim[i, j + 1] = np.sqrt(np.dot(coefs, yyy))
                resid_sim[i, j + 1] = sigma_sim[i, j + 1] * shocks[i, j]
                y_sim[i, j] = mean + resid_sim[i, j + 1]
        return y_sim, resid_sim[:, 1:], sigma_sim[:, 1:]

    def get_single_index_startvals(self, start_vals: Sequence) -> tuple[float, float]:
        start_date, quantile = start_vals
        fc = self.garch_fit[0].forecast(horizon=1, start=start_date, reindex=False).variance
        dfstart = pd.concat([self.residuals, self.conditional_vol], axis=1).loc[start_date:]
        idx = fc[fc.iloc[:, 0] == fc.iloc[:, 0].quantile(quantile, interpolation="higher")].index
        idx = pd.to_datetime(idx).strftime("%Y-%m-%d")[0]
        start = dfstart[idx:idx].values.squeeze()
        return start[0], start[1]

    def get_multi_index_startvals(self, start_vals: Sequence) -> tuple[np.ndarray, np.ndarray]:
        start_date, quantile = start_vals
        sigma_step1 = np.empty_like(self.training_data_scaled[start_date:])
        for i in range(len(self.tickers)):
            sigma_step1[:, i] = np.sqrt(
                self.garch_fit[i].forecast(horizon=1, start=start_date, reindex=False)
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

    def custom_nday_price_simulation(
        self,
        horizon1: int,
        horizon2: int,
        ndays: int,
        shock_type: str,
        nsims: int,
        VaRp: float = 0.99,
        start_vals: Sequence | None = None,
        bootstrap_start: str | None = None,
        bootstrap_end: str | None = None,
    ) -> dict:
        """Return the legacy wide dictionary of GARCH VaR results."""

        horizon = horizon1 + horizon2

        if shock_type == "normal":
            shocks = self.get_normal_shocks(horizon, nsims)
        elif shock_type == "custom_normal":
            shocks = self.get_custom_normal_shocks(horizon1, horizon2, nsims)
        elif shock_type == "bootstrap":
            if bootstrap_start is None or bootstrap_end is None:
                raise ValueError("bootstrap_start and bootstrap_end are required for bootstrap shocks")
            shocks = self.get_bootstrap_shocks(horizon, nsims, bootstrap_start, bootstrap_end)
        elif shock_type == "custom_bootstrap":
            if bootstrap_start is None or bootstrap_end is None:
                raise ValueError("bootstrap_start and bootstrap_end are required for bootstrap shocks")
            shocks = self.get_custom_boot_shocks(
                horizon1,
                horizon2,
                nsims,
                bootstrap_start,
                bootstrap_end,
            )
        else:
            raise ValueError(
                "shock_type must be one of: normal, custom_normal, bootstrap, custom_bootstrap"
            )

        if self.single_index:
            coefs = np.array(self.coefficients[0]).squeeze()
            mean = np.array(self.constant_mean[0])
            if start_vals is None:
                resid_start = self.residuals.iloc[-1][0]
                sigma_start = self.conditional_vol.iloc[-1][0]
            else:
                resid_start, sigma_start = self.get_single_index_startvals(start_vals)

            ysims, _, _ = self.nday_price_sim(
                horizon=horizon,
                shocks=shocks,
                coefs=coefs,
                mean=mean,
                resid_start=resid_start,
                sigma_start=sigma_start,
                nsims=nsims,
            )
            self.sim_paths = ysims / self.yscale[0]
        else:
            if start_vals is not None:
                resid_start_values, sigma_start_values = self.get_multi_index_startvals(start_vals)

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
                    resid_start = resid_start_values[i]
                    sigma_start = sigma_start_values[i]

                ysims[:, :, i], residsims[:, :, i], sigmasims[:, :, i] = self.nday_price_sim(
                    horizon=horizon,
                    shocks=shocks[:, :, i],
                    coefs=coefs,
                    mean=mean,
                    resid_start=resid_start,
                    sigma_start=sigma_start,
                    nsims=nsims,
                )
                ysims[:, :, i] = ysims[:, :, i] / self.yscale[i]

            self.sim_paths = np.empty(shape=(nsims, horizon))
            for i in range(nsims):
                self.sim_paths[i, :] = ysims[i, :, :].dot(self.weights)

        sim_paths_sub = self.sim_paths[:, horizon1:]
        ndaypc = pd.DataFrame(sim_paths_sub.T).rolling(ndays).sum().dropna().reset_index(drop=True)
        var = ndaypc.quantile(1 - VaRp, interpolation="higher") * 100

        if self.single_index:
            legacy_weight_description = "1"
        else:
            legacy_weight_description = ";".join(str(x) for x in self.weights)

        return {
            "Tickers": self.ticker_description,
            "Weights": legacy_weight_description,
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "BootstrapStart": bootstrap_start if shock_type in ["bootstrap", "custom_bootstrap"] else None,
            "BootstrapEnd": bootstrap_end if shock_type in ["bootstrap", "custom_bootstrap"] else None,
            "ShockType": shock_type,
            "VaRp": str(round(VaRp * 100)) + "Q",
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
            "ConvergencyCode": ";".join(str(x) for x in self.convergence_codes),
            "LogVaR_50": round(var.quantile(1 - 0.50, interpolation="higher") * -1, 4),
            "LogVaR_75": round(var.quantile(1 - 0.75, interpolation="higher") * -1, 4),
            "LogVaR_95": round(var.quantile(1 - 0.95, interpolation="higher") * -1, 4),
            "LogVaR_99": round(var.quantile(1 - 0.99, interpolation="higher") * -1, 4),
            "VaR_50": round(-(np.exp(var.quantile(1 - 0.5, interpolation="higher") / 100) - 1) * 100, 4),
            "VaR_75": round(-(np.exp(var.quantile(1 - 0.75, interpolation="higher") / 100) - 1) * 100, 4),
            "VaR_95": round(-(np.exp(var.quantile(1 - 0.95, interpolation="higher") / 100) - 1) * 100, 4),
            "VaR_99": round(-(np.exp(var.quantile(1 - 0.99, interpolation="higher") / 100) - 1) * 100, 4),
        }

    # Backward-compatible alias for current notebooks and regression harnesses.
    CustomNDayPriceSimulation = custom_nday_price_simulation

