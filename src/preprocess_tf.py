# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:50:59 2020

@author: gilbe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:08:33 2020

@author: gilbe
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
tf.random.set_seed(13)


def train_val_tf(x_train,
                 y_train,
                 x_val,
                 y_val,
                 x_test,
                 y_test,
                 batch_size=128,
                 buffer_size=10000):
    """
    Transform data to tensorflow format.
    """

    BATCH_SIZE = batch_size
    BUFFER_SIZE = buffer_size

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.cache().batch(BATCH_SIZE).repeat()

    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val = val.batch(BATCH_SIZE).repeat()

    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test = test.batch(BATCH_SIZE).repeat()

    return train, val, test


def time_series_split(df, train_size=0.6, val_size=0.2, test_size=0.2):
    """Split series in train, test and validation sets.

    --Args:
        * df: dataframe with all data.
        * train_size: train set size in percent of total df length
        * val_size: validation size in percent of total df length
        * test_size: test size in percent of total df length

    TODO
    ---
    1) return train val and test index
    2) index will be used in to tensor to return tensor index

    """
    check_sum = None
    N = len(df)
    if val_size is not None and test_size is not None:
        check_sum = train_size + val_size + test_size

        if check_sum == float(round(1)):

            train_size = round(N * train_size)
            df_train = df.iloc[:train_size]

            vs = round(train_size + N * val_size)
            df_val = df.iloc[train_size: vs]

            ts = round(train_size + val_size * N + test_size * N)
            df_test = df.iloc[vs:]

        else:
            print(f'Check sum: {check_sum}')
    return df_train, df_val, df_test


def add_datefeatures(df):
    try:
        df.index = pd.to_datetime(df.index)
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        # df['dayofmonth'] = df.index.dayofmonth
        df['dayofyear'] = df.index.dayofyear
        df['year'] = df.index.dayofyear
        return df
    except Exception as e:
        print('An exception occurred:', e)


def add_cyclic_datepart(df, cols=None):
    """Apply the sine function to date features.


    Args:
    ---
    * df: dataframe with date index.
    * columns: specify date columns.


    Return
    ---
    Return only the date columns
    """
    if cols is None:
        cols = ['day', 'dayofweek', 'month', 'quarter',
                'dayofyear', 'year']
    new_cols = [c for c in df.columns if c in cols]
    df[cols] = df[cols].apply(lambda x: np.sin(x))
    return df[cols]


def helper_train_test(data_x, data_y, lookback):
    """Helper function for the creation of time series data"""
    x, y = [], []

    time_length = len(data_x)

    for i in range(time_length - lookback):
        x.append(data_x[i: i + lookback])
        y.append(data_y[i + lookback])

    return np.array(x), np.array(y)


def to_tensor(data,
              date_features=None,
              add_cyclic_date=False,
              lookback=30,
              transformer_x=None,
              use_transformer=False,
              # return_time_idx=False,
              rolling_split=False,
              verbose=False):
    """
    Transform inputs to 3-D tensors
    and y as the one time step ahead.


    Args:
        * data: data to create time series targets and features for
        LSTM.
        * lookback:

    ---
    Shape of data:
        features: (total trading days, history for regression, no of features)
        labels: (total trading days, no of features)

    Return:
    """

    if add_cyclic_date:
        x = np.concatenate((data, date_features), axis=1)
    # just do a copy of same data
    else:
        x = data
    y = data

    # repeat this for train-val-test
    xtrain, xval, ytrain, yval = train_test_split(
        x, y, shuffle=False, random_state=42, test_size=0.25)

    xval, xtest, yval, ytest = train_test_split(
        xval, yval, shuffle=False, random_state=42, test_size=0.5)

    imputer_x = SimpleImputer(strategy='median')
    xtrain = imputer_x.fit_transform(xtrain)
    xval = imputer_x.transform(xval)
    xtest = imputer_x.transform(xtest)

    imputer_y = SimpleImputer(strategy='median')
    ytrain = imputer_y.fit_transform(ytrain)
    yval = imputer_y.transform(yval)
    ytest = imputer_y.transform(ytest)

    if use_transformer:
        if transformer_x is None:
            transformer_x = MinMaxScaler()
            xtrain = transformer_x.fit_transform(xtrain)
            xval = transformer_x.transform(xval)
            xtest = transformer_x.transform(xtest)

    xtrain, ytrain = helper_train_test(xtrain, ytrain, lookback)
    xval, yval = helper_train_test(xval, yval, lookback)
    xtest, ytest = helper_train_test(xtest, ytest, lookback)

    return xtrain, xval, xtest, ytrain, yval, ytest


def data_pipe(df,
              transformer_x=None,
              use_transformer=False,
              # return_time_idx=True,
              use_tf_data=False,
              add_cyclic_date=False):
    """Data pipe splits data in train-val-test, then
    it does preprocessing on it. This logic might be implemented
    on the layes itself.

    Args:
        * df: dataframe with data to train.
        * transformer: use a data transformer for
        preprocessing.
        * use_tf_data: if True use the tf-data-set class.
    ---


    Return
    ---
    A dictionary with each of train, val and test sets.

    """


    """ split should be done in the to_tensor function """
    # df_train, df_val, df_test = time_series_split(df)
    # TODO: get time indexes here
    # concat those inside the to_tensor function
    # transform to sine
    if add_cyclic_date:
        data_datefeatures = add_cyclic_datepart(add_datefeatures(df))

        xtrain, xval, xtest, ytrain, yval, ytest = to_tensor(
            df, data_datefeatures, use_transformer=use_transformer)
    else:
        xtrain, xval, xtest, ytrain, yval, ytest = to_tensor(
            df, use_transformer=use_transformer)

    if use_tf_data:
        data_train, data_val, data_test = train_val_tf(
            xtrain, ytrain, xval, yval, xtest, ytest)

        return dict(data_train=data_train,
                    data_val=data_val,
                    data_test=data_test)

    return dict(xtrain=xtrain, ytrain=ytrain,
                xval=xval, yval=yval,
                xtest=xtest, ytest=ytest)


# def create_rolling_ts(
#     input_data, 
#     lookback=5, 
#     return_target=True,
#     apply_datefeatures=True,
#     # return_array=False,
#     **kwargs
#     ):
#     """
#     Make flat data by using pd.concat instead, pd.concat([df1, df2]).
#     Slow function.
#     Save data as preprocessed?
#     """
#     x = []
#     y = []
#     rows = len(input_data)
#     features = input_data.copy()
#     target = input_data.copy()
#     for i in range(rows - lookback):
#         """Create embeddings for the date-features"""
#         if apply_datefeatures:
#             rolling_features = date_features(features.iloc[i: i + lookback])
#         else:
#             rolling_features = features.iloc[i: i + lookback]

#         rolling_target = target.iloc[i + lookback: i + lookback + 1]
#         x.append(rolling_features)
#         y.append(rolling_target)
#     # if return_array:
#     x = np.array(x)
#     y = np.array(y)

#     if return_target:
#         return x, y
#     return x


# def split_data(data, train_size, valid_size):
#     """
#     Implement data based splitting. 
#     Do normalization.
#     """
#     train_size = int(len(data) * train_size)
#     valid_size = int(train_size + len(data) * valid_size)
#     try:
#         train_set = data.iloc[: train_size]
#         valid_set = data.iloc[train_size: valid_size]
#         test_set = data.iloc[valid_size: ]
#         return train_set, valid_set, test_set
#     except Exception as e:
#         print(f'Exception from _split_data: {e}')


# def new_function():
#     pass


# def factory():
#     return True


# def date_features(df):
#     if isinstance(df, pd.core.series.Series):
#         df = pd.DataFrame(df, index=df.index)

#     df.loc[:, 'day_of_year'] = df.index.dayofyear
#     df.loc[:, 'month'] = df.index.month
#     df.loc[:, 'day_of_week'] = df.index.day
#     df.loc[:, 'hour'] = df.index.hour
#     return df