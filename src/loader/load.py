import yfinance as yf
import seaborn as sns
import pandas as pd
import warnings
import requests 
import getpass
import csv
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class YFinanceDataset:
    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')

    def __init__(
        self, 
        # ticker, 
        start_date='2020-08-16', 
        end_date='2021-12-12', 
        interval='1d',
        col='Close'
        ):
        # self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.col = col

    def get_single_ticker(self, ticker):
        try:
            ticker_data = yf.Ticker(ticker) #get ticker data
            ticker_df = ticker_data.history(
                interval=self.interval, 
                start=self.start_date,
                end=self.end_date, 
                back_adjust=True
                ) 
            df_data = ticker_df[self.col]         
            return df_data
        except Exception as e:
            print(f"Exception: {e}. Allowed columns: ['Open', 'High', 'Low', 'Close', 'Volume']")

    def get_multiple_tickers(
        self, 
        ticker_names
        ):

        features_df = pd.DataFrame()

        for ticker in ticker_names:
            df_data = self.get_single_ticker(ticker)
            df_data = pd.DataFrame(df_data, index=df_data.index)
            df_data.columns = [f'{ticker}_{c}' for c in df_data.columns] 
            features_df = pd.concat([features_df, df_data], axis=1) #concatenate tickers and their features
        return features_df

    def get_single_fundamentals(self):
        """TODO: ADD FUNDAMENTALS FOR SINGLE AND MANY STOCKS."""
        pass