# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 04:02:46 2019

@author: gilbe
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor # used for transforming the target variable
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
np.random.seed(100)


PROJECT_ROOT_DIR = os.path.join(os.path.dirname('__file__'), '.')
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'kaggle')

"""
Best set of parameters for Random Forest Regressor
-------------------------------------------------
{'bootstrap': True,
 'criterion': 'mae',
 'max_depth': 7,
 'max_features': 5,
 'min_samples_split': 10,
 'n_estimators': 62}



#    if param_dist is None and model is RandomForestRegressor:
#        param_dist = {"max_depth": sp_randint(5, 10),
#                      "max_features": sp_randint(2, 8),
#                      "min_samples_split": sp_randint(5, 10),
#                      "bootstrap": [True, False],
#                      "criterion": ["mae", "mse"], 
#                      "n_estimators": sp_randint(60, 70)}

"""

# if pushing code to git paste this to 


def load_data():
    """ This function returns xtrain and test sets"""
    train = pd.read_csv(os.path.join(PROJECT_DATA_DIR, 
                                     'train_winton.csv')).drop('Id', axis=1)
    test = pd.read_csv(os.path.join(PROJECT_DATA_DIR,
                                    'test_2_winton.csv')).drop('Id', axis=1)

    print('Done Loading data ...')
    return train, test


def plot_stocks(xtrain, row=1, colstart=2, endcol=None):
    plt.figure(figsize=(10, 6))
    plt.title('Minute stock returns')
    plt.plot(xtrain.iloc[:row, colstart:endcol].values.flatten())
    plt.xlabel('time in minutes')
    plt.ylabel('returns in (minutes)');


def get_cols(xtrain, seach_str='Feature'):
    cols = 0
    for col in xtrain.columns:
        if col.startswith('Feature'):
            cols += 1
    print("{} columns contained the search word {} ".format(cols, seach_str))
    return cols


def get_xy(data, col=146):
    x = data.iloc[:, :col]
    y = data.iloc[:, col:]
    return x, y


def delete_features(data):
    cols = get_cols(data)
    return data.iloc[:, cols:]


def pre_process(x, y, scaler=QuantileTransformer):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, shuffle=True)
    
    imp = SimpleImputer(strategy='mean')
    xtrain = imp.fit_transform(xtrain)
    xtest = imp.transform(xtest)
    
    # set some condition?
    scaler = scaler(output_distribution='normal')
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    # dim reduction
    pca = PCA()
    pca.fit(xtrain)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    
    pca = PCA(n_components=d)
    xtrain = pca.fit_transform(xtrain)
    xtest = pca.transform(xtest)

    return xtrain, xtest, ytrain, ytest


def hyper_parameter_search(xtrain,
                           ytrain,
                           xtest, 
                           ytest, 
                           model,
                           param_dist=None, 
                           train_samples=5000, 
                           test_samples=1000,
                           n_iter_search=10,
                           cv=5):
    
#    transformer = QuantileTransformer(output_distribution='normal')
    # we might need to do grid or random search
    regressor = model()
    print('model used: {}'.format(repr(regressor)))
    print('')
    
#    reg = TransformedTargetRegressor(regressor=regressor, 
#                                     transformer=transformer)

    n_iter_search = n_iter_search
    reg = RandomizedSearchCV(regressor, 
                             param_distributions=param_dist, 
                             n_iter=n_iter_search, 
                             cv=cv, 
                             iid=False, 
                             verbose=10, 
                             n_jobs=-1)

    # use the train examples only
    # for hyper parameter optimization
    n_train_samples = len(xtrain)
    n_test_samples = len(xtest)
    
    xtr, ytr, xts, yts = None, None, None, None
    
    if (train_samples < n_train_samples) and (test_samples < n_test_samples):
        xtr = xtrain[:train_samples, :].copy()
        ytr = ytrain[:train_samples].copy()

        xts = xtest[:test_samples, :].copy()
        yts = ytest[:test_samples].copy()
    else:        
        xtr = xtrain.copy()
        ytr = ytrain.copy()
        
        xts = xtest.copy()
        yts = ytest.copy()

    reg.fit(xtr, ytr)
    print('Done with training.')
    pred_train = reg.predict(xtr)
    pred_test = reg.predict(xts)

    sctrain = r2_score(ytr, pred_train)
    sctest = r2_score(yts, pred_test)
    
    mse_train = mean_squared_error(ytr, pred_train)
    mse_test = mean_squared_error(yts, pred_test)
    
    ex_variance_train = explained_variance_score(ytr, pred_train)
    ex_variance_test = explained_variance_score(yts, pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    s1 = 'R2 score train set: {} score test set: {}'
    s2 = 'Mean-squared error train set: {} test set: {}'
    s3 = 'Explained variance train set: {} test set: {}'
    s4 = 'Root mean squared error train set {} test set: {}'
    print(s1.format(round(sctrain, 4), round(sctest, 4)))
    print(s2.format(round(mse_train, 4), round(mse_test, 4)))
    print(s3.format(round(ex_variance_train, 4), round(ex_variance_test, 4)))
    print(s4.format(round(rmse_train, 4), round(rmse_test, 4)))
    print('.' * 50)
    print('Best parameters:')
    print(reg.best_params_)
    
    return reg, reg.best_params_

def plot_feature_importance():
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == '__main__':
    # load data
    train, test = load_data()
    xtrain, ytrain = get_xy(train)
    y1 = ytrain['Ret_PlusOne'] # D + 1 returns
    y2 = ytrain['Ret_PlusTwo'] # D + 2 returns
    cols = get_cols(xtrain)
    # cols starts delete cols with features
    xtrain = xtrain.iloc[:, cols: ]
    xtrain1, xtest1, ytrain1, ytest1 = pre_process(xtrain, y1)
    
    param_dist = {"max_depth": sp_randint(2, 10), 
                  "max_features": ['auto', 'sqrt', 'log2', None], 
                  "min_samples_split": sp_randint(2, 50), 
                  'min_samples_leaf': sp_randint(2, 500),
                  "n_estimators": sp_randint(200, 300), 
                  'loss': ['huber', 'lad', 'lad'], 
                  'learning_rate': [0.1, 0.2, 0.3, 0.0001], 
                  'n_iter_no_change': sp_randint(1, 50)}
    
    train_samples = 15000
    test_samples = 5000
    
    hyper_parameter_search(xtrain1, 
                           ytrain1, 
                           xtest1, 
                           ytest1, 
                           model=GradientBoostingRegressor,
                           param_dist=param_dist,
                           n_iter_search=5,
                           train_samples=train_samples,
                           test_samples=test_samples,
                           cv=5)
    
    # save preprocessed data to disk