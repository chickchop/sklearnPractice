'''
Created on 2017. 5. 8.

@author: ko
'''
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def main():
    #import data
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                       'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    print('Class lables', np.unique(df_wine['Class label']))
    print(df_wine.head())
    
    #split data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #feature Scailing(normalization)
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    
    #feature Scailing(Standardization)
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)
    
    #avoiding overfitting = penalty
    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    print('Training Accuracy:', lr.score(X_train_std, y_train))
    print('Test Accuracy:', lr.score(X_test_std, y_test))
    
    
def ploting():
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
    weights, params = [], []
    
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1', C=10**c, random_state=)

if __name__ == '__main__':
    main()