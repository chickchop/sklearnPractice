'''
Created on 2017. 7. 8.

@author: ko-mac
'''
from sklearn.linear_model.base import LinearRegression
from sklearn.metrics.regression import r2_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_regression():
    X = np.array([258.0, 270.4, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
    
    lr = LinearRegression()
    pr = LinearRegression()
    
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)
    
    #simple regression fitting
    lr.fit(X, y)
    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)
    
    #polynomial regression fitting
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    
    #plotting
    plt.scatter(X, y, label='training points')
    plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='quadratic fit')
    plt.legend(loc='upper left')
    plt.show()
    
def hounsing_example():
    #data import
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    
    X = df[['LSTAT']].values
    y = df[['MEDV']].values  
    regr = LinearRegression()
    
    #create polynomial features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)
    
    #linear fit
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))   
    
    #quadratic fit
    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))
    
    #cubic fit
    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))
    
    #plot results
    plt.scatter(X, y, label='training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
    plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
    plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $10000 \'s [MEDV]')
    plt.legend(loc='upper right')
    plt.show()
    
    #transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)
    
    #fit features
    X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
    
    #plot results
    plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
    plt.xlabel('% log(lower status of the population [LSTAT])')
    plt.ylabel('\squrt{Price \; in \; \$10000 \'s [MEDV]$')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    compare_regression()
    hounsing_example()