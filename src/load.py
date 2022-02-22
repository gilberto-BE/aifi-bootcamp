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


def from_list_df(price_list):
    """
    Create dataframe from list from 
    data_alpha_vantage.
    """
    time_col = [col[0] for col in price_list[1:]]
    open_col = [col[1] for col in price_list[1:]]
    high_col = [col[2] for col in price_list[1:]]
    low_col = [col[3] for col in price_list[1:]]
    close_col = [col[4] for col in price_list[1:]]
    vol_col = [col[5] for col in price_list[1:]]

    df = pd.DataFrame(
        {
        'open': open_col, 
        'high': high_col,
        'low': low_col,
        'close': close_col,
        'volume': vol_col
        }, index=time_col
        )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.astype(float)
    return df


def data_alpha_vantage(
    key=None, 
    ticket='IBM', 
    interval='15min&slice=year1month12'
    ):
    """Get data from alpha vantage. Return list with
    time, open, high, low, close, volume."""
    try:
        function='TIME_SERIES_INTRADAY_EXTENDED'
        ROOT_URL = 'https://www.alphavantage.co/query?'
        url = f'{ROOT_URL}function={function}&symbol={ticket}&interval={interval}&apikey={key}'
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            price_list = list(cr)
        return price_list

    except ConnectionError:
        raise ConnectionError("Connection error, you need to pass key.")


def dataframe_from_list(price_list):
    """Convert price series from 
    list to dataframe. Input comes 
    from data_alpha_vantage."""
    return from_list_df(price_list)


class IntradayExtended:
    def __init__(
        self, 
        key, 
        ticket='AAPL', 
        interval='15min&slice=year1month12'
        ):

        self.key = key
        self.ticket = ticket
        self.interval = interval

    def get_intraday_extended(self):
        df = data_alpha_vantage(self.key, self.ticket, self.interval)
        df = from_list_df(df)
        return df


class TimeSeriesLoader:
    """
    Data loader of time series. Based on the 
    Alpha Vantage API. This class mostly transform data 
    to pandas dataframes.

    Supported values for interval:
    * 1min, 5min, 15min, 30min, 60min
    """
    def __init__(
        self, 
        symbol, 
        function='TIME_SERIES_INTRADAY', 
        interval=5, 
        ):

        self.symbol = symbol
        self.source = "https://www.alphavantage.co/query?"
        self.function = function
        self.interval = interval
        self.apikey = self.set_apikey()

    def set_apikey(self):
        try:
            apikey = getpass.getpass(prompt="Enter your apikey: ")
            return apikey
        except getpass.GetPassWarning:
            raise ValueError("You have to enter an apikey.")

    def get_json(self, url):
        """Pass the relevant url"""
        json_data = requests.get(url)
        data = json_data.json()
        return data
        
    def ts_intraday(self, interval=None):
        url = f'{self.source}function={self.function}&symbol={self.symbol}&interval={self.interval}min&apikey={self.apikey}&adjusted=true&outputsize=full'
        data = self.get_json(url)
        df = pd.DataFrame.from_dict(data[f'Time Series ({self.interval}min)']).T
        return df

    def ts_day_adjusted(self):
        try:
            func = 'TIME_SERIES_DAILY_ADJUSTED'
            url = f'{self.source}function={func}&symbol={self.symbol}&apikey={self.apikey}'
            data = self.get_json(url)
            df = pd.DataFrame.from_dict(data['Time Series (Daily)']).T
            return df
        except:
            raise ConnectionError("time series adjusted is a paid service.")

    def ts_intraday_extended(self, interval):
        df = IntradayExtended(
            self.apikey, 
            self.symbol, 
            interval
            ).get_intraday_extended()
        return df


