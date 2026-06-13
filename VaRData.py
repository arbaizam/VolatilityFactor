
import pandas as pd
import numpy as np
from functools import cached_property
from pathlib import Path
from constants import CONFIGS, F_DATA

 
#----------------------------------------------------------------#
#------------------------GLOBAL VARS-----------------------------#
#----------------------------------------------------------------#

#fpath = CONFIGS["data_settings"]["loc_pricing_data"]

DATA = pd.read_csv(F_DATA, parse_dates = ["Date"], 
    dtype = {"Price": np.float64}, thousands = ",").sort_values(by=["Ticker", "Date"])


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#----------------------------------------------------------------#

class VarData:
    """
    A class for processing and storing historical price data of a list of tickers.

    Attributes:
        tickers (list): A list of strings representing the tickers for which the price data is to be retrieved.
        weights (list, optional): A list of floats representing the weights of each ticker in the portfolio. Default is None, which assumes equal weighting.

    Properties:
        tickers (getter, setter): A getter and setter method for tickers attribute, allowing assignment of tickers as a string, tuple or list. 
            Invalidates the weights property if assigned a new value.
        weights (getter, setter): A getter and setter method for weights attribute, allowing assignment of weights as a list. If the tickers
            list has only one element, weights are automatically set to 1.0. If the weights are not provided, they are automatically set to equal weighting across all tickers. Raises an exception if the sum of the weights is not equal to one.

    Cached Properties:
        single_index: A boolean that indicates whether the tickers list has only one element.
        price_matrix: A pandas dataframe that contains the historical price data of the tickers.
        price_return_matrix: A pandas dataframe that contains the one-day percentage change of the tickers' prices.
        wtd_price_return: A pandas dataframe that contains the weighted one-day percentage change of the tickers' prices.
        log_price_matrix: A pandas dataframe that contains the log of the tickers' historical prices.
        log_return_matrix: A pandas dataframe that contains the one-day log change of the tickers' prices.
        wtd_log_return: A pandas dataframe that contains the weighted one-day log change of the tickers' prices.
        index_description: A string (semi-colon delimited) of the tickers' index price description (from vendor).
        ticker_description: A string (semi-colon delimited) description of the ticker's index (from vendor).
        volatility_factor_id: A string for the mda-assigned ID.

    Methods:
        None

    """
    def __init__(self, tickers, weights=None):
        self.tickers = tickers
        self.weights = weights

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, value):
        self._weights = None #invalidate property
        if isinstance(value, tuple): #tuple-to-list
            self._tickers = list(value)
        elif isinstance(value, str): #single-value to list
            self._tickers= [value]
        elif isinstance(value,list): #list
            self._tickers= value
        else: 
            raise ValueError(f"{value} is of object type {type(value)}.  Valid types include: str, tuple, or list")

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if len(self.tickers) ==1:
            self._weights = np.array(1)
        else:
            if value is None:
                self._weights = np.array([1/len(self.tickers)] * len(self.tickers))
            else:
                self._weights = np.array(value)
                if sum(self._weights) != 1:
                    raise Exception(f"Weights must add to one.  Weights totaled {sum(self._weights)}")

    @cached_property
    def single_index(self):
        return True if len(self.tickers) == 1 else False

    @cached_property
    def price_matrix(self):
        df = DATA[DATA["Ticker"].isin(self.tickers)].copy().reset_index(drop=True)
        df = pd.DataFrame(df.pivot(index = "Date", values = "Price", columns = "Ticker").rename_axis(None, axis=1))[self.tickers]
        return df

    @cached_property
    def price_return_matrix(self):
        """
        Returns:
            pd.Dataframe: one-day percentage change
        """
        return (self.price_matrix.diff()/ self.price_matrix.shift(1)).dropna()

    @cached_property
    def wtd_price_return(self):
        """
        Returns:
            pd.Dataframe: weighted one-day percentage change
        """
        return pd.DataFrame(data = np.dot(self.price_return_matrix, self.weights), 
                            columns = ["Wtd_Price_Return"], 
                            index = self.price_return_matrix.index)

    @cached_property
    def wtd_price(self):
        """
        Returns:
            pd.Dataframe: weighted price
        """
        return pd.DataFrame(data = np.dot(self.price_matrix, self.weights), 
                            columns = ["Wtd_Price"], 
                            index = self.price_matrix.index)
    
    @cached_property
    def log_price_matrix(self):
        """
        Returns:
            pd.Dataframe: log prices
        """      
        return np.log(self.price_matrix)

    @cached_property
    def log_return_matrix(self):
        """
        Returns:
            pd.Dataframe: one-day log return
        """          
        return self.log_price_matrix.diff().dropna()

    @cached_property
    def wtd_log_return(self):
        """
        Returns:
            pd.Dataframe: weighted one-day log returns
        """         
        return pd.DataFrame(data = np.dot(self.log_return_matrix, self.weights),
                            columns = ["Wtd_Log_Return"],
                            index = self.log_return_matrix.index)

    @cached_property
    def index_description(self):
        """
        Returns:
            str: semi-colon delimited description of index
        """             
        df = DATA[DATA["Ticker"].isin(self.tickers)].copy().reset_index(drop=True)
        df = df.groupby("Ticker")["Description"].first()
        vals = []
        for i in range(df.shape[0]):
            vals.append(f"{df.iloc[i]} ({df.index[i]})")
        return str(vals).replace("['","").replace("']", "").replace(",", ";")

    @cached_property
    def ticker_description(self):
        """
        Returns:
            str: semi-colon delimited tickers
        """   
        return ";".join([str(tick) for tick in self.tickers])
    
    @cached_property
    def volatility_factor_id(self):
        """
        Returns:
            str: volatility_factor_id (uses self.tickers to lookup values from CONFIGS)
        """   

        for idx, vals in CONFIGS["var_settings"].items():
            if vals["tickers"] == self.tickers:
                vf_id = idx
        if "vf_id" not in locals():
            print(f"Unknown Volatility Factor ID for {self.tickers}")
            vf_id = "Unknown"


        return vf_id