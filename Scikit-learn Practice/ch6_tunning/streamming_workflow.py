'''
Created on 2017. 7. 3.

@author: ko
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def streamming_workflow():
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
    print('Test Accuracy: %.3f' %pipe_lr.score(X_test, y_test))


if __name__ == '__main__':
    streamming_workflow()