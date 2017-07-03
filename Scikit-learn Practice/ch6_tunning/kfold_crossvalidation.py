'''
Created on 2017. 7. 3.

@author: ko
'''
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


def k_fold_crossvalidation():
    #data import
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    
    #data labeling
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M','B'])
    
    #data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    #standardization and data compression, finally testing
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    
    #k-fold validation
    kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.:%s, Acc:%.3f' %(k+1, np.bincount(y_train[train]), score))
    
    print('1.')    
    print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
    
    #use package
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
    
    print('\n2.')
    print('CV accuracy scores: %s' %scores)
    print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

if __name__ == '__main__':
    k_fold_crossvalidation()