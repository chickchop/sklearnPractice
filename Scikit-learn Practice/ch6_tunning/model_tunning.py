'''
Created on 2017. 7. 3.

@author: ko
'''
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_hyperparameter_combination():
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
    
    #create streaming work flow pipeline (standardization, learning)
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
    
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C' : param_range, 'clf__kernel' : ['linear']}, {'clf__C' : param_range, 'clf__gamma' : param_range, 'clf__kernel' : ['rbf']}]
    
    #search best hyperparameter combination
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
    
    gs = gs.fit(X_train, y_train)
    
    print(gs.best_score_)
    
    clf= gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: %.3f' %clf.score(X_test, y_test))
    
    
if __name__ == '__main__':
    find_hyperparameter_combination()