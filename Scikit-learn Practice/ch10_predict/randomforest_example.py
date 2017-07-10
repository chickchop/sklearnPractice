'''
Created on 2017. 7. 10.

@author: ko
'''
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import mean_squared_error, r2_score


def housing_randomforest_example():
    #data import
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    
    X = df.iloc[:, :-1].values
    y = df[['MEDV']].values  
    
    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
    
    #fitting
    forest.fit(X_train, y_train)
    
    #predict
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    #result
    print('MSE train: %.3f, test: %.3f' %(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test:%.3f' %(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    
    #plot results
    plt.scatter(y_train_pred, y_train_pred - y_train, label='training data', c='black', marker='o', s=35, alpha=0.5)
    plt.scatter(y_test_pred, y_test_pred - y_test, label='test data', c='lightgreen', marker='s', s=35, alpha=0.7)
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.show()

if __name__ == '__main__':
    housing_randomforest_example()