from VaRData import VarData
import pandas as pd
import numpy as np
from arch import arch_model
import re
from numba import jit
from typing import Tuple, List, Optional


class GarchSim(VarData):
    """A class to encapsulate GARCH volatility models and simulation processes

    Args:
        tickers (List): A list of index pricing tickers.
        weights (numpy array): An array of weights for each ticker.
        sample_start (str): The start date of the training data.
        sample_end (str): The end date of the training data.
        mean_model (str, optional): The mean model to use. Defaults to "Zero".
        vol_model (str, optional): The volatility model to use. Defaults to "Garch".

    Attributes:
        sample_start (str): The start date of the training data.
        sample_end (str): The end date of the training data.
        mean_model (str): The mean model to use.
        vol_model (str): The volatility model to use.
        training_data (pandas.DataFrame): The training data.
        training_data_scaled (pandas.DataFrame): The training data scaled by a factor.
        garch_fit (list): A list of return objects from arch.fit method.
        yscale (list): A list of scaling factors.
        convergence_codes (list): A list of convergence codes.
        conditional_vol (pandas.DataFrame): A DataFrame of conditional volatility estimates.
        residuals (pandas.DataFrame): A DataFrame of residuals.
        std_residuals (pandas.DataFrame): A DataFrame of standardized residuals.
        coefficients (list): A list of model coefficients.
        constant_mean (list): A list of constant means.
        std_returns (pandas.DataFrame): A DataFrame of standardized returns.

    Raises:
        ValueError: If the sample_start or sample_end arguments are not formatted as "YYYY-MM-DD".

    Methods:
        _estimate_garch(): A private method to estimate the GARCH model for the given training data.
        get_normal_shocks():  Method called from CustomNDayPriceSimulation(); returns n-d array shocks (normal dist)
        get_custom_normal_shocks(): Method called from CustomNDayPriceSimulation(); returns n-d array shocks (normal dist) w/shock
        get_bootstrap_shocks(): Method called from CustomNDayPriceSimulation(); returns n-d array fitted shocks
        get_custom_boot_shocks(): Method called from CustomNDayPriceSimulation(); returns n-d array fitted shocks AND normal
        nday_price_sim(): Method called from CustomNDayPriceSimulation(); algorithim for daily price forecast
        get_single_index_starvals(): Method to return starting values for use in single-index simulation process.
        get_multi_index_startvals(): Method to return starting values for use in multi-index simulation process.
        CustomNDayPriceSimulation(): Method to simulate n-day price forecast.
    """
    def __init__(self, tickers: List, weights: np.array, sample_start:str, sample_end:str, 
                    mean_model: Optional[str] = "Zero", vol_model: Optional[str] = "Garch"):
        super().__init__(tickers, weights)
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.mean_model = mean_model
        self.vol_model = vol_model
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
    def sample_start(self):
        return self._sample_start

    @sample_start.setter
    def sample_start(self, value):
        self._training_data = None #invalidate cached values
        self._training_data_scaled = None
        self._garch_fit = None
        self._yscale = None
        self._convergence_codes = None
        if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            self._sample_start = value
        else:
            raise ValueError("sample_start must be formatted as string (YYYY-MM-DD)")

    @property
    def sample_end(self):
        return self._sample_end

    @sample_end.setter
    def sample_end(self, value):
        if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            self._sample_end = value
        else:
            raise ValueError("sample_start must be formatted as string (YYYY-MM-DD)")           

    @property
    def training_data(self):
        if self._training_data is None:
            self._estimate_garch()
        return self._training_data      

    @property
    def training_data_scaled(self):
        if self._training_data_scaled is None:
            self._estimate_garch()
        return self._training_data_scaled   

    @property
    def garch_fit(self):
        if self._garch_fit is None:
            self._estimate_garch()
        return self._garch_fit

    @property
    def yscale(self):
        if self._yscale is None:
            self._estimate_garch()
        return self._yscale

    @property
    def convergence_codes(self):
        if self._convergence_codes is None:
            self._estimate_garch()
        return self._convergence_codes

    @property
    def conditional_vol(self):
        if self._conditional_vol is None:
            self._estimate_garch()
        return self._conditional_vol

    @property
    def residuals(self):
        if self._residuals is None:
            self._estimate_garch()
        return self._residuals

    @property
    def std_residuals(self):
        if self._std_residuals is None:
            self._estimate_garch()
        return self._std_residuals

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._estimate_garch()
        return self._coefficients       

    @property
    def constant_mean(self):
        if self._constant_mean is None:
            self._estimate_garch()
        return self._constant_mean    

    @property
    def std_returns(self):
        if self._std_returns is None:
            self._estimate_garch()
        return self._std_returns   

    def _estimate_garch(self):
        """Estimates the GARCH model parameters and fits the model to the training data.

        This method first selects a subset of the log returns data as the training data based on the specified `sample_start` 
        and `sample_end` attributes. Then, it scales the training data by different factors and applies the GARCH(1,1) model 
        to each column in the scaled data to estimate the model parameters. If the estimation fails to converge, 
        the method tries again with a different scaling factor. Finally, it saves the estimated parameters, 
        residuals, and other related information as class attributes for later use.

        Returns:
        None
        """
        self._training_data = None
        self._training_data_scaled = None
        self._garch_fit = None
        self._yscale = None
        self._convergence_codes = None
        self._conditional_vol = None
        self._residuals = None
        self._std_residuals = None
        self._coefficients = None
        self._constant_mean = None
        self._std_returns = None 
        TrainingData = self.log_return_matrix[self.sample_start:self.sample_end].copy()
        TrainingDataScaled = pd.DataFrame(data = None, columns = TrainingData.columns, index = TrainingData.index)
        Residuals = pd.DataFrame(data = None, columns = TrainingData.columns, index = TrainingData.index)
        StdResiduals = pd.DataFrame(data = None, columns = TrainingData.columns, index = TrainingData.index)
        ConditionalVol = pd.DataFrame(data = None, columns = TrainingData.columns, index = TrainingData.index)
        Coefficients = []
        Constant_Mean = []
        Y_Scales = [] 
        Garch_Fit = [] 
        ConvergenceCodes = [] 
        for col in TrainingData.columns:
            iters = 0
            convergency_flag = None
            while iters < 4:
                if iters == 0:
                    y_scale = 1000
                elif iters == 1:
                    y_scale = 10000
                elif iters == 2:
                    y_scale = 100
                elif iters == 3:
                    y_scale = 10
                training_data_scaled_temp = TrainingData[col] * y_scale
                garch_model = arch_model(y= training_data_scaled_temp, 
                                        mean=self.mean_model, 
                                        vol=self.vol_model, 
                                        p=1, o=0, q=1, 
                                        dist="Normal", 
                                        rescale=False) 
                garch_fit = garch_model.fit(disp="off", show_warning=False)
                convergency_flag = int(garch_fit.convergence_flag)
                if convergency_flag == 0:
                    break
                iters +=1
            TrainingDataScaled[col] = training_data_scaled_temp
            Residuals[col] = garch_fit.resid.values
            StdResiduals[col] = garch_fit.std_resid.values 
            ConditionalVol[col] = garch_fit.conditional_volatility.values
            Coefficients.append([garch_fit.params["omega"], garch_fit.params["alpha[1]"], garch_fit.params["beta[1]"]])
            if garch_fit.num_params == 4:
                Constant_Mean.append(garch_fit.params["mu"])
            else:
                Constant_Mean.append(0)
            Garch_Fit.append(garch_fit)
            Y_Scales.append(y_scale)
            ConvergenceCodes.append(convergency_flag)
        self._training_data = TrainingData
        self._training_data_scaled = TrainingDataScaled
        self._yscale = Y_Scales
        self._garch_fit = Garch_Fit
        self._convergence_codes = ConvergenceCodes
        self._residuals = Residuals
        self._std_residuals = StdResiduals
        self._coefficients = Coefficients
        self._constant_mean = Constant_Mean
        self._std_returns = TrainingDataScaled / ConditionalVol
        self._conditional_vol = ConditionalVol

    def get_normal_shocks(self, horizon:int, nsims:int)-> np.ndarray:
        """n-d array of shocks to draw from for price simulation.
        Args:
            horizon: number of days to simulate.  Used to construct arrays equal to horizon shape.
            nsims: Number of paths to simulate. Used to construct nd arrays.
        Returns:
            nd array of shocks for random draws in custom price simulation
        """
        random_state = np.random.RandomState(1)
        if self.single_index is True:
            shocks = random_state.standard_normal(size=(nsims,horizon))
        else: 
            sigma = self.std_returns.corr().values
            shocks = random_state.multivariate_normal(mean=np.zeros(len(self.tickers)), cov=sigma, size=(nsims,horizon))
        return shocks

    def get_custom_normal_shocks(self, horizon1:int, horizon2:int, nsims:int)-> np.ndarray:
        """n-d array of shocks to draw from for price simulation.
        Args:
            horizon1: first componet of total days to simulate.  Used to construct arrays equal to horizon shape
            horizon2: second componet of total days to simulate.  Used to construct arrays equal to horizon shape
            nsims: Number of paths to simulate. Used to construct nd arrays
        Returns:
            nd array of shocks for random draws in custom price simulation
        """
        horizon = horizon1 + horizon2
        shocks = self.get_normal_shocks(horizon, nsims)
        shocks[:,horizon1:(horizon1 + 5)] *= np.sqrt(2)
        return shocks

    def get_bootstrap_shocks(self, horizon:int, nsims:int, bootstrap_start:str, bootstrap_end:str)-> np.ndarray:
        """nd array of fitted shocks for price simulation
        Args:
            horizon: total days to simulate.  Used to construct arrays equal to horizon shape.
            nsims: Number of paths to simulate. Used to construct nd arrays.
            bootstrap_start: start date for bootstrap
            bootstrap_end: end date for bootstrap
        Returns:
            nd array of shocks for random draws in custom price simulation
        """
        random_state = np.random.RandomState(1)
        if self.single_index is True:
            fitted_shocks = self.std_residuals
            boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
            shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims, horizon))
            shocks = boot_shocks[shock_idx]
        else:
            fitted_shocks = self.std_returns
            boot_shocks = np.asarray(fitted_shocks[bootstrap_start:bootstrap_end]).squeeze()
            shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims,horizon))
            shocks = np.empty(shape=(nsims, horizon, len(self.tickers)))
            for i in range(nsims):
                shocks[i,:,:] = boot_shocks[shock_idx[i]]
        return shocks


    def get_custom_boot_shocks(self, horizon1:int, horizon2:int, nsims:int, bootstrap_start:str, bootstrap_end:str)-> np.ndarray:
        """n-d array of shocks to draw from for price simulation.
        Args:
            horizon1: horizon1 of the forecast.  Used to construct arrays equal to horizon1 shape.
            horizon2: horizon2 of the forecast.  Used to construct arrays equal to horizon2_forecast shape. 
            n_sims: Number of price paths to simulate. Used to construct nd arrays.
            bootstrap_start: For fitted shocks start
            bootstrap_end: For fitted shocks end 
        Returns:
            nd array of shocks for random draws in custom price simulation
        """
        random_state = np.random.RandomState(1)
        if self.single_index is True:
            fitted_shocks = self.std_residuals #fitted shocks
            h2shocks = random_state.standard_normal(size=(nsims, horizon2)) #horizon2 shocks
        else:
            fitted_shocks = self.std_returns #fitted shocks
            sigma = fitted_shocks.corr().values
            h2shocks = random_state.multivariate_normal(mean=np.zeros(fitted_shocks.shape[1]), cov=sigma, size=(nsims, horizon2)) #horizon2 shocks
        boot_shocks = np.asarray(fitted_shocks[bootstrap_start: bootstrap_end]).squeeze()  #horizon1 shocks (bootstrapped from fitted_shocks)
        shock_idx = random_state.randint(0, boot_shocks.shape[0], size=(nsims,horizon1))  
        h1shocks = boot_shocks[shock_idx]
           
        return np.hstack((h1shocks, h2shocks)) 

    @staticmethod
    @jit(nopython=True)
    def nday_price_sim(horizon: int, shocks: np.ndarray, coefs: np.array, mean: np.array, resid_start:float, 
            sigma_start: float, nsims: int)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """returns three nd arrays for the simulated residuals, sigma, and yvalues
        Args:
            horizon: Number of periods to simulate
            shocks: nday array of shocks to use in simulation
            coefs: array of coefficients (omega, alpha, beta)
            mean: single element array for mean.
            resid_start: starting residual value
            sigma_start: starting sigma value
            nsims: number of simulations 

        Returns:
            Tuple of nd arrays (nsims x horizon) for y, residuals, sigma
        """
        resid_sim = np.empty(shape = (nsims, horizon + 1), dtype=np.float64) 
        sigma_sim = np.empty(shape = (nsims, horizon + 1), dtype=np.float64) 
        y_sim = np.empty(shape = (nsims, horizon), dtype=np.float64) 

        for i in range(np.int64(nsims)):
            for j in range(horizon):
                resid_sim[i,0] = resid_start
                sigma_sim[i,0] = sigma_start
                yyy = np.array([1, resid_sim[i,j]**2, sigma_sim[i,j]**2])
                sigma_sim[i, j + 1] = np.sqrt(np.dot(coefs, yyy))
                resid_sim[i, j + 1] = sigma_sim[i,j + 1] * shocks[i,j]
                y_sim[i,j] = mean + resid_sim[i,j + 1]
        return y_sim, resid_sim[:,1:], sigma_sim[:,1:]

    def get_single_index_startvals(self, start_vals: List)-> Tuple[float]:
        """ 
        Gets starting values for use in single-index simulation process.

        Args:
            start_vals: two-element list of date and quantile to use to
                derive starting residual and sigma in simulation process.

        Returns:
            Tuple of starting residual and sigma values.
        """
        start_date, quantile = start_vals
        fc = self.garch_fit[0].forecast(horizon=1, start=start_date, reindex=False).variance 
        dfstart = pd.concat([self.residuals,self.conditional_vol],axis=1).loc[start_date:]
        idx = fc[fc.iloc[:,0] == fc.iloc[:,0].quantile(quantile, interpolation = "higher")].index
        idx = pd.to_datetime(idx).strftime("%Y-%m-%d")[0]
        start = dfstart[idx:idx].values.squeeze()
        resid_start = start[0]
        sigma_start = start[1]
        return resid_start, sigma_start

    def get_multi_index_startvals(self, start_vals: List)-> Tuple[float]:
        """ 
        Gets starting values for use in multi-index simulation process.

        Args:
            start_vals: two-element list of date and quantile to use to
                deriving starting residual and sigma in simulation process.
        """
        start_date, quantile = start_vals
        #step 1: one-step sim
        sigma_step1 = np.empty_like(self.training_data_scaled[start_date:])
        for i in range(len(self.tickers)):
            sigma_step1[:,i] = np.sqrt(self.garch_fit[i].forecast(horizon=1, start=start_date, reindex=False).variance.values.squeeze())
        Rho = self.std_residuals.corr().values
        #step 2: weighted standard deviation of portfolio
        portfolio_sigma = []
        for i in range(len(sigma_step1)):
            M_sigma = np.diag(sigma_step1[i])
            Sigma2 = M_sigma.dot(Rho).dot(M_sigma)
            sigma2_weight = self.weights.dot(Sigma2).dot(self.weights)
            sigma_weight = np.sqrt(sigma2_weight)
            portfolio_sigma.append(sigma_weight)
        sigma_p = pd.Series(data = portfolio_sigma, index = self.training_data[start_date:].index)
        idx = sigma_p[sigma_p == sigma_p.quantile(quantile, interpolation = "higher")].index
        idx = pd.to_datetime(idx).strftime("%Y-%m-%d")[0]
        dfstart = pd.concat([self.residuals,self.conditional_vol],axis=1).loc[start_date:,]
        start = dfstart[idx:idx].values.squeeze()
        start = np.transpose(start.reshape(2,len(self.tickers)))
        resid_start = start[:,0]
        sigma_start = start[:,1]
        return resid_start, sigma_start


    def CustomNDayPriceSimulation(self, horizon1: int, horizon2:int, ndays:int, shock_type:str, nsims:int, VaRp: Optional[float] = 0.99,
            start_vals: Optional[List] = None, bootstrap_start: Optional[List] = None, bootstrap_end: Optional[List] = None) -> dict:
        
        """returns dictionary of VaR results

        Args:
            horizon1: represents lag (number of period dropped from simulation)
            horizon2: periods to forecast
            ndays: number of days in liquidation period
            shock_type: one of "custom", "custom_normal", "bootstrap", "custom_bootstrap"
            nsims: number of simulations 
            start_vals: To set the starting residual and sigma values for the simulation,
                pass a list to this parameter as follows: [date, quantile]; i.e. start_vals = ["2021-10-01", 0.5].  
                If set to None (default value), the starting resid/sigma used in the simulation default to the last values 
                in the sample period.  Defaults to None.
            bootstrap_start: start date (YYYY-DD-MM) for bootstrap scenarios. Required for shock_type
                equal to bootstrap or custom_bootstrap.  Defaults to None.
            bootstrap_end: end date (YYYY-DD-MM) for bootstrap scenarios. Required for shock_type
                equal to bootstrap or custom_bootstrap.  Defaults to None
        Returns:
            dict: results of simulation and related meta-data
        """
        if self.garch_fit is None:
            self._estimate_garch()
  
        horizon = horizon1 + horizon2

        if shock_type == "normal":
            shocks = self.get_normal_shocks(horizon, nsims)
        elif shock_type == "custom_normal":
            shocks = self.get_custom_normal_shocks(horizon1, horizon2, nsims)
        elif shock_type == "bootstrap":
            shocks = self.get_bootstrap_shocks(horizon, nsims, bootstrap_start, bootstrap_end)
        elif shock_type == "custom_bootstrap":
            shocks = self.get_custom_boot_shocks(horizon1, horizon2, nsims, bootstrap_start, bootstrap_end)
        else:
            raise ValueError("shock_type must be one of the following: normal, custom_normal, bootstrap, custom_bootstrap")
        
        if self.single_index:
            coefs = np.array(self.coefficients[0]).squeeze()
            mean = np.array(self.constant_mean[0])
            if start_vals is None:
                resid_start = self.residuals.iloc[-1][0]
                sigma_start = self.conditional_vol.iloc[-1][0]
            else:
                resid_start, sigma_start = self.get_single_index_startvals(start_vals)
            ysims,residsims,sigmasims = self.nday_price_sim(horizon = horizon, shocks=shocks, coefs=coefs, mean = mean, 
                                            resid_start=resid_start, sigma_start=sigma_start, nsims=nsims)
            self.sim_paths = ysims / self.yscale[0] 
        else:
            if start_vals is not None:
                Resid_Start, Sigma_Start = self.get_multi_index_startvals(start_vals)
            ysims = np.empty_like(shocks)
            residsims = np.empty_like(shocks)
            sigmasims = np.empty_like(shocks)
            for i in range(len(self.tickers)):
                coefs = np.array(self.coefficients[i]).squeeze()
                mean = np.array(self.constant_mean[i])
                if start_vals is None:
                    resid_start = self.residuals.iloc[-1,i]
                    sigma_start = self.conditional_vol.iloc[-1,i]
                else:
                    resid_start = Resid_Start[i]
                    sigma_start = Sigma_Start[i]
                ysims[:,:,i], residsims[:,:,i], sigmasims[:,:,i] = self.nday_price_sim(horizon = horizon, shocks=shocks[:,:,i], coefs=coefs, 
                                                                        mean = mean, resid_start=resid_start, sigma_start=sigma_start, nsims=nsims)
                ysims[:,:,i] = ysims[:,:,i] / self.yscale[i] 
            self.sim_paths = np.empty(shape=(nsims, horizon))
            for i in range(nsims):
                self.sim_paths[i,:] = ysims[i,:,:].dot(self.weights)
        
        sim_paths_sub = self.sim_paths[:,horizon1:] 
        ndaypc = pd.DataFrame(sim_paths_sub.T).rolling(ndays).sum().dropna().reset_index(drop=True)
        var = ndaypc.quantile(1-VaRp, interpolation = "higher") * 100 #leave as decimal


        return_dict = {
            "Tickers": self.ticker_description,
            "Weights": ";".join([str(x) for x in self.weights]) if self.single_index is False else str(self.weights),
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "BootstrapStart": bootstrap_start if shock_type in ["bootstrap", "custom_bootstrap"] else None,
            "BootstrapEnd": bootstrap_end if shock_type in ["bootstrap", "custom_bootstrap"] else None,
            "ShockType": shock_type,
            "VaRp": str(round(VaRp * 100)) + "Q",
            "MeanModel": self.mean_model,
            "VolatilityModel": self.vol_model,
            "Method": f"GarchSim_{shock_type}_{str(round(start_vals[1]*100))}" if start_vals != None else f"GarchSim_{shock_type}",
            "Horizon1": horizon1,
            "Horizon2": horizon2,
            "NDays": ndays,
            "Simulations": nsims,
            "StartingValueQ": str(round(start_vals[1]*100)) + "Q" if start_vals is not None else "None",
            "StartingValueDate": start_vals[0] if start_vals is not None else "None",
            "ConvergencyCode": ";".join([str(x) for x in self.convergence_codes]),
            "LogVaR_50": round(var.quantile(1-0.50, interpolation="higher") * - 1, 4),
            "LogVaR_75": round(var.quantile(1-0.75, interpolation="higher") * - 1, 4),
            "LogVaR_95": round(var.quantile(1-0.95, interpolation="higher") * - 1, 4),
            "LogVaR_99": round(var.quantile(1-0.99, interpolation="higher") * - 1, 4),
            "VaR_50": round(-(np.exp(var.quantile(1-0.5, interpolation="higher")/100)-1)*100, 4),
            "VaR_75": round(-(np.exp(var.quantile(1-0.75, interpolation="higher")/100)-1)*100, 4),
            "VaR_95": round(-(np.exp(var.quantile(1-0.95, interpolation="higher")/100)-1)*100, 4),
            "VaR_99": round(-(np.exp(var.quantile(1-0.99, interpolation="higher")/100)-1)*100, 4)}
            
        return return_dict
    

class NonParametricVar(VarData):
    def __init__(self, tickers: list, weights:np.array, sample_start: str, sample_end: str):
        self.sample_start = sample_start
        self.sample_end = sample_end
        self._training_data = None
        super().__init__(tickers, weights)

    @property
    def sample_start(self):
        return self._sample_start

    @sample_start.setter
    def sample_start(self, value):
        self._training_data = None #invalidate cached values
        if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            self._sample_start = value
        else:
            raise ValueError("sample_start must be formatted as string (YYYY-MM-DD)")
        
    @property
    def sample_end(self):
        return self._sample_end

    @sample_end.setter
    def sample_end(self, value):
        if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            self._sample_end = value
        else:
            raise ValueError("sample_start must be formatted as string (YYYY-MM-DD)")     

    @property
    def training_data(self):
        if self._training_data is None:
            self._training_data = self.wtd_log_return[self.sample_start: self.sample_end]
        return self._training_data      
    
    @staticmethod
    def _get_boot_quantile(boot_i: np.ndarray, n_liquidation, VaRp):
        """
            Returns the VaR at a given confidence level (VaRp) for a given bootstramp sample (boot_i)
            Attributes:
            1. boot_i: n-dimensional array of bootstramp sample. 
            2. n_liquidation: number of days 
            3. VaRp: confidence level
            """
        df = pd.DataFrame(boot_i)
        pc_nday = df.rolling(n_liquidation).sum().dropna()
        var = pc_nday.quantile(q = 1 - VaRp, interpolation ='higher')
        var = 1 - np.exp(var)
        return(var)

    def CircularBlockBootstrap(self, n_liquidation: int, nboot: int = 1000, VaRp: float = 0.99) -> dict:
        """
            Return VaRs based on the Circular Block Bootstrap model
            Attributes:
            1. n_liquidation: the liquidation horizon
            2. nboot: the number of bootstrap replications
            3. VaRp: the confidence level of the VaR
        """

        # Step 2: Create blocks for the Block Circular Bootstrap
        #   The blocks are determined by the data sample and the block size. For example, suppose the data sample
        #   is [1,2,3,4,5,6,7,8,9,10], and the block size is 4. Then we have 10 (equals to the sample size) blocks as:
        #   [1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7], [5,6,7,8], 
        #   [6,7,8,9], [7,8,9,10], [8,9,10,11], [9,10,11,12], [10,11,12,13]  

        sample = np.append(self.training_data, self.training_data)
        sample_size = len(self.training_data)                       # number of obs. of each bootstrap sample
        block_size = int(np.floor(np.sqrt(sample_size)))            # number of obs. in each block
        block_n = int(np.ceil(sample_size / block_size))            # number of blocks needed to create a bootstrap sample
        
        # create the pool of blocks 
        sample_block = np.empty([sample_size, block_size])
        for i in range(sample_size):
            sample_block[i,] = sample[i : i + block_size]

        # Step 3: Create nboot(=1000 in our case) different bootstrap sample
        #   a. Let's continue the above sample to create a bootstrap sample. First, the bootstrap sample size equals to 
        #   the data sample size, here it is 10. Because the each block has 4 data points, we need 3 block to create a 
        #   bootstrap sample. Let's random draw 3 blocks from the 10 blocks defined above, say they are [6,7,8,9], [1,2,3,4]
        #   and [9,10,11,12]. Then the bootstrap sample is [6,7,8,9,1,2,3,4,9,10]. Note that we get rid of the last two data 
        #   points because the sample size is 10.
        #   b. We repeat the above process nboot times, so we will have nboot different bootstrap samples 
        random_state = np.random.RandomState(1)
        boot_index = random_state.randint(sample_size, size = (nboot, block_n))
        boot_all = sample_block[boot_index]
        boot_array = boot_all.reshape((nboot,(boot_all.shape[1] * boot_all.shape[2])))[:, :sample_size]

        # Step 4: Implement the 'boot_quantile' function to get the VaR for each bootstrap sample
        quantile_boot_all = np.apply_along_axis(self._get_boot_quantile, axis=1, arr=boot_array, n_liquidation = n_liquidation, VaRp = VaRp)
        quantile_boot_mean = quantile_boot_all.mean() * 100
        
        return_dict = {
            "Tickers": self.ticker_description,
            "Weights": ";".join([str(x) for x in self.weights]) if self.single_index is False else str(self.weights),
            "Method": "CircularBootstrap",
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "VaRp": str(round(VaRp * 100)) + "Q",
            "NDays": n_liquidation,
            "BootstrapReplications": nboot,
            "VaR": quantile_boot_mean,
        }

        return return_dict