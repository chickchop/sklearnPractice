'''
Created on 2017. 7. 7.

@author: ko
'''
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    #data import
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    
    X = df[['RM']].values
    y = df[['MEDV']].values
    
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                             residual_threshold=5.0, random_state=0)
    
    #fitting
    ransac.fit(X, y)
    
    #plotting
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    
    plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper left')
    plt.show()
    
if __name__ == '__main__':
    main()