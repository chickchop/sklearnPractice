'''
Created on 2017. 7. 4.

@author: ko
'''
from scipy import interp
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats_basic import threshold


def predict_confusion_matrix():
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
    
    #fitting
    pipe_svc.fit(X_train, y_train)
    
    #predict
    y_pred = pipe_svc.predict(X_test)
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    
    print('Precision: %.3f' %precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' %recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    
    
def predict_roc_auc():
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
    
    X_train2 = X_train[:, [4, 14]]
    
    #cross-validation
    cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
    
    fig = plt.figure(figsize=(7,5))
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:,1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        roc_acu = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' %(i+1, roc_acu))
        
    plt.plot([0,1], [0,1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
    
    plt.plot([0,0,1], [0,1,1], lw=2, linestyle=':', color='black', label='perfect performance')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
if __name__ == '__main__':
    predict_confusion_matrix()
    predict_roc_auc()