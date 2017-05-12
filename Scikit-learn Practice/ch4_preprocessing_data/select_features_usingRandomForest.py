'''
Created on 2017. 5. 12.

@author: ko
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def main():
    #import data
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                       'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    feat_labels = df_wine.columns[1:]
    
    #split data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
    #generate random forest classifier
    forest = RandomForestClassifier(n_estimators=10000, random_state=0,n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    #print feature importance
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" %(f + 1, 30, feat_labels[f], importances[indices[f]]))
    
    #plot feature importance
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
    plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()