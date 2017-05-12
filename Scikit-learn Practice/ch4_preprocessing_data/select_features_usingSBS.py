'''
Created on 2017. 5. 11.

@author: ko
'''
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing.data import StandardScaler
from SBS import SBS

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
    
    #feature Scaling(Standardization)
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)
    
    #classifier generate
    knn = KNeighborsClassifier(n_neighbors=2)
    
    #checking classifier performance
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    
    #plot performance
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()
    
    #show good features
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])

    #estimate knn classifier
    knn.fit(X_train_std, y_train)
    print('Training accuracy:', knn.score(X_train_std, y_train))
    print('Training accuracy:', knn.score(X_test_std, y_test))
    
    #re-estimate knn classifier
    knn.fit(X_train_std[:, k5], y_train)
    print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
    print('Training accuracy:', knn.score(X_test_std[:, k5], y_test))
if __name__ == '__main__':
    main()