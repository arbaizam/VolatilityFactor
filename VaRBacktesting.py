from VaRData import VarData
import pandas as pd
import numpy as np
import re
from functools import cached_property
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


class VarBacktest(VarData):
    def __init__(self, tickers, n_liquidation=None, weights = None, start_date = None, end_date = None, VaR = None, 
        VaR_is_log = True, forward_shift = False):
        """
        Args:
            tickers (str): n-charater ticker associated with index.
            n_liquidation (int, optional): number of days in liquidation period
            weights (list, optional): portfolio weights. Defaults to None.
            start_date (str, optional): YYYY-MM-DD. Defaults to None.  If set to None, starts at first date available. 
            end_date (str, optional): YYYY-MM-DD. Defaults to None.  If set to None, end at last date available.
            VaR (float, optional): Value to backtest.  For VaR of 2.0%, enter 2.0.
            VaR_is_log (bool, optional): Set to True if the value for VaR is the logrithim.  If set to True, VaR will be 
            transformed to a percentage.
            forward_shift (bool, optional): Set to True to reverse the calculation of changes.  Defaults to note.
                Example: consider this scenario:
                n_liquidation = 1; 1-1-2000 price = 100; 1-2-2000 price = 105 (5% one-day percent change)
                if forward_shift = False; the 5% percentage change will be reflected on 1-2-2000;
                if forward_shift = True; the 5% percentage change will be reflected on 1-1-2000;
        """
        super().__init__(tickers, weights)
        self.start_date = start_date
        self.end_date = end_date
        self.n_liquidation = n_liquidation
        self.VaR = VaR
        self.VaR_is_log = VaR_is_log
        self.weights = weights
        self.forward_shift = forward_shift
    
    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        if value is None:
            self._start_date = value
        else:
            if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
                self._start_date = value
            else:
                raise ValueError("start_date must be None or formatted date (YYYY-MM-DD)")

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, value):
        if value is None:
            self._end_date = value
        else:
            if re.match("^[\d]{4}-[\d]{2}-[\d]{2}$", value):
                self._end_date = value
            else:
                raise ValueError("end_date must be None or formatted date (YYYY-MM-DD)")
 
    @cached_property
    def var_pct(self):
        """Scalar: transforms log to arthmetic return if Var_is_log is true; otherwise
        returns var / 100
        """
        if self.VaR_is_log is True:
            var = -(np.exp(-self.VaR / 100) -1)
        else:
            var = self.VaR / 100
        return var

    @cached_property
    def nday_price_return(self):
        """pd.Series of n-day percentage change; if multiple-index, 
        if multi-index, returns weighted percentage change
        """
        df = self.price_matrix.sort_index().copy()

        if self.forward_shift is False:
            pct_chg = df / df.shift(self.n_liquidation) - 1
        else:
            pct_chg = df.shift(-self.n_liquidation) / df - 1

        if self.single_index is True:
            pct_chg = pd.Series(data = pct_chg.iloc[:,0].values, index = df.index)
        else:
            pct_chg = pd.Series(data = pct_chg.dot(self.weights), index = df.index)
        return pct_chg[self.start_date: self.end_date].dropna()

    @cached_property
    def nday_log_return(self):
        """pd.Series of n-day percentage change; if multiple-index, 
        if multi-index, returns weighted percentage change
        """
        df = self.log_price_matrix.sort_index().copy()

        if self.forward_shift is False:
            log_chg = df - df.shift(self.n_liquidation)
        else:
            log_chg = df.shift(-self.n_liquidation) - df
    
        if self.single_index is True:
            log_chg = pd.Series(data = log_chg.iloc[:,0].values, index = df.index)
        else:
            log_chg = pd.Series(data = log_chg.dot(self.weights), index = df.index)
        return log_chg[self.start_date: self.end_date].dropna()
    
    @cached_property
    def break_series(self):
        """ pd.Series of breaks; where 1 = Break and 0 = No Break
        """
        return (-self.var_pct > self.nday_price_return) * 1

    @cached_property
    def shortfall_series(self):
        """ pd.Series of shortfalls; negative values = breaks (shortfall) and
        positive values = no break (longfall)
        """
        return self.nday_price_return + self.var_pct 

    @cached_property
    def duration(self):
        """list of duration of each break (in days)
        """
        breaks = np.array(self.break_series)
        breaks = np.insert(breaks, 0, 1) #insert star val
        breaks = np.append(breaks, 1) #insert end val
        return list(np.diff(np.where(breaks == 1))[0,1:-1])#drop start/end vals

    def backtest_plot(self, title_text = "Standard", show_plot = True, file_name=None):
        """ Creates backtesting plot; blue = positive return, red = negative return,
        red "x" = break, dotted line = VaR 
        Note: if file_name is NOT None, plot will be saved to file_name
        """
        sns.set_context("paper")
        sns.set_theme(style = "whitegrid", font_scale = 1.15) 

        r = self.nday_price_return
        q = pd.Series(data = -self.var_pct, index=self.nday_price_return.index)

        ax = r[r <= q].plot(color="red", marker="x", ls="None", figsize=(18, 8.5)) # breaks
        for h in r[r <= q].index:
            plt.axvline(h, color="gray", alpha=0.4, linewidth=1, zorder=0)
        r[r > 0].plot(ax=ax, color="#93B1CC", marker="o", ls="None") # Positive returns
        r[(q <= r) & (r <= 0)].plot(ax=ax, color="#DA9694", marker="o", ls="None") # Negative returns but no hit
        q.plot(ax=ax, grid=False, color="#003359", rot=0, ls="dashed") # VaR

        # Title / Axes
        if title_text == "Standard":
            title_text = f"{self.index_description}: {self.n_liquidation}-Day Backtesting Results"
        else:
            title_text = title_text
        plt.title(f"{title_text}")
        plt.xlabel("")
        plt.ylabel(f"{self.n_liquidation}-Day Percent Change")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0, decimals=2))
        ax.yaxis.grid()
        ax.set_xlim(self.start_date, self.end_date)

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")

        if show_plot is True:
            plt.show()

        return None

    def price_change_plot(self, title_text = "Standard", weighted = True, show_plot = True, file_name=None):
        """ Creates plot of n-day price percentage change.  If weighted = True, will plot the weighted percentage change;
        if weighted = False, will plot each index as a single series.  
        Note: if file_name is NOT None, plot will be saved to file_name
        """
        if weighted is True:
            ax = self.nday_price_return.plot(color = "#003359", figsize=(18, 8.5))
        else:
            df = self.price_matrix.sort_index().copy()
            pct_chg = (df / df.shift(self.n_liquidation) - 1).dropna()
            ax = pct_chg[self.start_date: self.end_date].plot(color = ["#003359", "#BCB800"], figsize=(18, 8.5))   

        # Title / Axes
        if title_text == "Standard":
            title_text = f"{self.index_description}: {self.n_liquidation}-Day Percentage Change"
        else:
            title_text = title_text
        plt.title(f"{title_text}")
        plt.ylabel(f"{self.n_liquidation}-Day Percent Change")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0, decimals=2))

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")

        if show_plot is True:
            plt.show()

        return None

    def price_plot(self, title_text = "Standard", weighted = False, show_plot = True, file_name=None):
        """ Creates plot of prices.  If weighted = True, will plot the weighted price,
        if weighted = False, will plot each index as a single series.  
        Note: if file_name is NOT None, plot will be saved to file_name
        """
        df = self.price_matrix[self.start_date: self.end_date].sort_index().copy()
 
        if weighted is True:
            ser = pd.Series(data = df.values.dot(self.weights), index = df.index)
            ax = ser.plot(color = "#003359",figsize=(18, 8.5))
        else:
            ser = df
            ax = ser.plot(color = ["#003359", "#BCB800"], figsize=(18, 8.5))
        #Title
        if title_text == "Standard": 
            title_text = f"{self.index_description}: Price"
        else:
            title_text = title_text
        plt.title(f"{title_text}")
        plt.ylabel(f"Price")

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight")

        if show_plot is True:
            plt.show()

        return None