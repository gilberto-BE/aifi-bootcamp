from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
    )
import numpy as np
from preprocess import is_pandas


def symmetric_mape(ytrue, ypred):
    if is_pandas(ytrue):
        ytrue = ytrue.to_numpy()
    if is_pandas(ypred):
        ypred = ypred.to_numpy()
    return np.sum(np.abs(ytrue - ypred))/(np.sum(ytrue + ypred))


def regression_metrics(ytrue, ypred):
    r2 = r2_score(ytrue, ypred)
    mse = mean_squared_error(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)
    smape = symmetric_mape(ytrue, ypred)

    return mse, r2, mae, smape

