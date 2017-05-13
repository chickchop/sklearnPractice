'''
Created on 2017. 5. 13.

@author: ko
'''
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def main():
    #import data
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    #split data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #standardization
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    
    #calculate mean vector
    np.set_printoptions(precision=4)
    mean_vecs = []
    
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
    #scaling discrete matrix
    d = 13 #number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    
    #calculate discrete matrix
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13
    S_B = np.zeros((d, d))
    
    for i, mean_vec in enumerate(mean_vecs):
        n = X[y==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    
    print('Between-class scatter matrix:%sx%s' % (S_B.shape[0], S_B.shape[1]))

if __name__ == '__main__':
    main()