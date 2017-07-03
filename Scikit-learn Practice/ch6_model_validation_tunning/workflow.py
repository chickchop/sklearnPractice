'''
Created on 2017. 6. 3.

@author: ko
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    #import
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    #split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

if __name__ == '__main__':
    main()
    