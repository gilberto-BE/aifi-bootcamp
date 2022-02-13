from loaders import loader


if __name__ == "__main__":
    data_loader = loader.TimeSeriesLoader(symbol="AAPL", interval=1)
    df = data_loader.ts_intraday()
    print(df.head())