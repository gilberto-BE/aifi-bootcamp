# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 03:50:04 2019

@author: gilbe
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)


def choose_scaler(xtrain, xtest, name='Standard'):
    scaler = None
    if name == 'Standard':
        scaler = StandardScaler()
    elif name == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = QuantileTransformer()

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    return xtrain, xtest

if __name__ == "__main__":
    data = load_breast_cancer()
    print(data['feature_names'])
    print('')
    print('.' * 50)
    x = data['data']
    y = data['target']
    print('inputs:')
    print(x)
    print('the target is:')
    print(y)
    print('shape of x:', x.shape)
    print('shape of y:', y.shape)
    
    """ DataFrames are going to be used only for statistics"""
    xdf = pd.DataFrame(x)
    ydf = pd.DataFrame(y)
    xdf.columns = data['feature_names']
    print('.' * 50)
    print(xdf.T.head(len(xdf.columns)))
    print(xdf.aggregate([np.mean, np.std, np.median]).T)
    print()
    print(ydf.aggregate([np.mean, np.std, np.median]).T)
    ydf.plot.hist(figsize=(10, 6), bins=50, 
                  title='Distribution of malignant and benign breast cancer')
    plt.show()
    print(ydf.head())
    print()
    fract_target = (ydf.apply(pd.Series.value_counts)/len(ydf) * 100 ).round(2)
    print('Fraction of targets are: {}'.format(fract_target))
    print('.' * 50)
    
    # step 1 split data in train and test, then transform data with a transformer, normal, or min max
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    print('shape of xtrain:', xtrain.shape)
    print('shape of ')
    
    xtrain, xtest = choose_scaler(xtrain, xtest, name='MiMax')
    print('the scaled values are:')
    print(xtrain)
    
    # step 2 train the model
    model = LogisticRegression(solver='lbfgs')
    model.fit(xtrain, ytrain)
    
    # step 3 make predictitions with test set
    y_pred = model.predict(xtest)
    
    # step 4 use confusion matrix and classification report
    print('.' * 50)
    print('Classification report:')
    print(classification_report(ytest, y_pred))
    print('')
    print('Confusion matrix:')
    print(confusion_matrix(ytest, y_pred))
    print('.' * 50)
    
    
    
    
    
    
    
    
    
    
    
    

    
    