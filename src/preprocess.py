import numpy as np
import pandas as pd


def create_rolling_ts(
    input_data, 
    lookback=5, 
    return_target=True,
    apply_datefeatures=True,
    return_np_array=False
    ):
    """
    Make flat data by using pd.concat instead, pd.concat([df1, df2]).
    Slow function.
    Save data as preprocessed?
    """
    x = []
    y = []
    rows = len(input_data)
    features = input_data.copy()
    target = input_data.copy()
    for i in range(rows - lookback):
        """Create embeddings for the date-features"""
        if apply_datefeatures:
            rolling_features = date_features(features.iloc[i: i + lookback])
        else:
            rolling_features = features.iloc[i: i + lookback]

        rolling_target = target.iloc[i + lookback: i + lookback + 1]
        x.append(rolling_features)
        y.append(rolling_target)
    if return_np_array:
        x = np.array(x)
        y = np.array(y)

    if return_target:
        return x, y
    return x


def date_features(df):
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df, index=df.index)

    df.loc[:, 'day_of_year'] = df.index.dayofyear
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_of_week'] = df.index.day
    df.loc[:, 'hour'] = df.index.hour
    return df


def split_data(data, train_size, valid_size):
    """
    Implement data based splitting. 
    Do normalization.
    
    """
    train_size = int(len(data) * train_size)
    valid_size = int(train_size + len(data) * valid_size)
    try:
        train_set = data.iloc[: train_size]
        valid_set = data.iloc[train_size: valid_size]
        test_set = data.iloc[valid_size: ]
        return train_set, valid_set, test_set
    except Exception as e:
        print(f'Exception from _split_data: {e}')

def square(i):
    return i ** 2


def flatten(data):
    """Make data n X 1 dimensional"""
    return data.reshape(data.shape[0], -1)


def is_pandas(df):
    return isinstance(df, (pd.core.frame.DataFrame, pd.core.series.Series))