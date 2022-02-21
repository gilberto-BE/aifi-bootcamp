import numpy as np
import pandas as pd
# from pytest import Instance
from transformers import AutoTokenizer
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import pandas_datareader as pdr
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
import sklearn.model_selection as sm
import matplotlib.pyplot as plt


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 


def flatten(data, num_cols=1):
    return np.array(data).reshape(-1, num_cols)


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


def rename_set_index(ts):
    ts['Unnamed: 0'] = pd.to_datetime(ts['Unnamed: 0'])
    ts.set_index('Unnamed: 0', inplace=True)
    ts.index.rename('date', inplace=True)
    return ts


def rolling_train_val():
    """
    Create rolling train-val tensors
    The function returns an adecuate 
    input for LSTM.
    """
    pass 


def rolling_merge():
    """
    Create rolling merge of different features
    to be use in pipeline for LSTM.
    """
    pass


def tokenize_function(data, col="text"):
    """ data is a dictionary???"""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    tokens =  tokenizer(data[col], padding="max_length", truncation=True)
    return tokens


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


def merge_features_lagged_target(
    features: pd.core.frame.DataFrame,
     lagged_target: pd.core.series.Series
     ):
     features = features.merge(
         lagged_target, left_index=True,
          right_index=True
          )
     return features


class ToTensor(object):
    def __call__(self, features, target):
        return {
            'features': torch.from_numpy(features), 
            'target': torch.from_numpy(target)
            }


class DataDebuger:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def check_shape(self):
        assert self.x.shape[0] > 0
        assert self.y.shape[0] > 0    

    def print_info(self):
        data = [self.x, self.y]
        for d in data:
            print(f'Head:')
            print(d.shape)
            print(d.head())


class TimeSeriesProc:
    def __init__(
        self, 
        data,
        num_data=None,
        cat_data=None,
        lookback=30, 
        train_size=0.75,
        valid_size=0.125, 
        normalize_data=True,
        merge_with=None,
        how='inner'
        ):
        """
        Return train, valid and test dataframes 
        which will be preprocessed in TimeSeriesDataset.

        TODO:
        1) Handle dates as limits. like end-train, end-val, end-test
        """
        self.data = data
        self.num_data = num_data
        self.cat_data = cat_data
        # self.stock_name = stock_name
        self.train_size = train_size
        self.valid_size = valid_size
        self.normalize_data = normalize_data
        self.lookback = lookback
        self._split_data()
        self.merge_with = merge_with
        self.how = how
        self.numeric_cols = ['close'] #self.xtrain[0].select_dtypes('float64')
        self.cat_cols = ['label', 'label', 'day_of_year', 'month', 'day_of_week', 'hour']

        self.xtrain, self.ytrain = self._rolling_ts(self.train_set, self.merge_with)
        self.xvalid, self.yvalid = self._rolling_ts(self.valid_set, self.merge_with)
        self.xtest, self.ytest = self._rolling_ts(self.test_set, self.merge_with)

        if self.normalize_data:
            self.init_stats_feat()
            self.init_stats_tgt()

            print('feature_train[0].shape', self.xtrain[0].shape)
            print()
            """Convert to list, self.xtrain, etc are lists from start."""
            # Create loop for repeating code
            self.xtrain, self.ytrain = self.norm_xy(self.xtrain, self.ytrain)
            self.xvalid, self.yvalid = self.norm_xy(self.xvalid, self.yvalid)
            self.xtest, self.ytest = self.norm_xy(self.xtest, self.ytest)

            data_debuger = DataDebuger(self.xtrain[0], self.ytrain[0])
            data_debuger.check_shape()
            data_debuger.print_info()


    def norm_xy(self, x, y):
        return self.normalize_feat_list(x), self.normalize_tgt_list(y)

    def _split_data(self):
        """
        Implement data based splitting. 
        Do normalization.
        
        """
        train_size = int(len(self.data) * self.train_size)
        valid_size = int(train_size + len(self.data) * self.valid_size)
        try:
            self.train_set = self.data.iloc[: train_size]
            self.valid_set = self.data.iloc[train_size: valid_size]
            self.test_set = self.data.iloc[valid_size: ]
        except Exception as e:
            print(f'Exception from _split_data: {e}')

    def _rolling_ts(self, data, merge_with):
        lagged_target, target = create_rolling_ts(
            data, 
            lookback=self.lookback,
            merge_with=merge_with,
            how=self.how
            )
        return lagged_target, target 

    def _concat_df(self, data):
        df_stats = pd.DataFrame()
        if isinstance(data, list):
            for d in data:
                if isinstance(d, pd.core.frame.DataFrame):
                    df_stats = pd.concat([df_stats, d[self.numeric_cols]])
            return df_stats

    def init_stats_feat(self):
        """
        Apply only on train dataset.
        If features=False then apply transform on target.
        """
        # if isinstance(data, list):
        print(self.xtrain)
        df_stats = self._concat_df(self.xtrain)
        self.mean_feat_train = np.mean(df_stats, axis=0) 
        self.std_feat_train = np.std(df_stats, axis=0)

    def init_stats_tgt(self):
        # if isinstance(data, list):
        df_stats = self._concat_df(self.ytrain)
        self.mean_tgt_train = np.mean(df_stats, axis=0)
        self.std_tgt_train = np.std(df_stats, axis=0)

    def normalize_feat(self, data):
        """        
        If features=False then apply transform on target.
        """
        data  = (data - self.mean_feat_train)/self.std_feat_train
        return data#[[self.stock_name, 'day_of_year', 'month', 'day_of_week', 'hour']]

    def normalize_tgt(self, data):
        data  = (data - self.mean_tgt_train)/self.std_tgt_train
        return data

    def normalize_feat_list(self, feat_list):
        """Do concatenation with categorical data."""
        return [self.normalize_feat(x) for x in feat_list]

    def normalize_tgt_list(self, tgt_list):

        return [self.normalize_tgt(y) for y in tgt_list]
            
    def inverse_norm(self, norm_data, features=True):
        if features:
            return self.std_feat_train * norm_data + self.mean_feat_train
        else:
            return self.std_tgt_train * norm_data + self.mean_tgt_train

    def get_data(self):
        return {
            'train_set': {
                "xtrain": np.array(self.xtrain),
                 "ytrain": np.array(self.ytrain)
                 }, 
            'valid_set': {
                "xvalid": np.array(self.xvalid), 
                "yvalid": np.array(self.yvalid)
                }, 
            'test_set': {
                "xtest": np.array(self.xtest), 
                "ytest": np.array(self.ytest)
                }
                }


class TimeSeriesDataset(Dataset):

    def __init__(
            self,
            features,
            target
            ):
        
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]

        """Return dict of dicts"""
        return {'features': torch.from_numpy(np.array(features)).float(), 
                'target': torch.from_numpy(np.array(target)).float()
            }


