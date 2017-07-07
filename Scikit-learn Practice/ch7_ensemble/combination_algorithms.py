'''
Created on 2017. 7. 5.

@author: ko
'''
import operator

from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals import six
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    '''
    A majority vote ensemble classifier
    '''


    def __init__(self, classifiers, vote='classlabel', weights=None):
        '''
        Parameter
        ----------------------------------------------
        classifiers : array-like, shape=[n_classifiers]
            Different classifiers for the ensemble
            
        vote : str, shape = {'classlabel, 'probability'}
            voting label, and probability
        
        weights : array-like, shape=[n_classifiers]
            optional. if a list of int or float values apreprovided, the classifires are weighted by importance
        '''
        
        self.classifiers = classifiers
        self.named_classifiers = {key : value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        '''
        Parameter
        ---------------------------------------------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Matrix of training sample
            
        y : array-like, shape=[n_samples]
        
        Returns
        ----------------------------------
        self : object
        '''
        
        #Use LabelEncoder to ensure class labels start
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers :
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
            
        return self
    
    def predict(self, X):
        '''
        Predict class labels for X
        
        Parameter
        -------------------------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
            
        Returns
        -----------------------------
        maj_vote : array-like, shape = [n_samples]
            Predicted class label
        '''
        
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
            
        #class label vote
        else:
            #collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x : np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
            
        return maj_vote
    
    def predict_proba(self, X):
        '''
        Predict class probabilities for X
            
        Parameters
        --------------------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Traing vectors
            
        Returns
        ---------------------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        '''
        
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        
        return avg_proba
    
    def get_params(self, deep=True):
        '''
        Get classifier parameter names for GridSearch
        '''
        
        #deep == False
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        
        #deep == True
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' %(name, key)] = value
                    
            return out

def check_each_classifier():
    #data import
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1,2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    #split data(train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    
    #check each classifiers performance
    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    
    print('10-fold cross validation:\n')
    
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, scoring='roc_auc', cv=10)
        
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]" %(scores.mean(), scores.std(), label))
    
def combination_algorithms():
    #data import
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1,2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    #split data(train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    
    #check each classifiers performance
    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    
    #combine classifier
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ['Majority Voting']
    
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, scoring='roc_auc', cv=10)
        
        print('Accuracy: %0.2f(+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))
        
    #plotting roc_auc curve
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-', '-.']
    
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        #assuming the label of the positive class is 1
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        
        fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        
        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' %(label, roc_auc))
        
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    #plotting
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
    
    for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
        clf.fit(X_train_std, y_train)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='^', s=50)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='red', marker='o', s=50)
        axarr[idx[0], idx[1]].set_title(tt)
        
    plt.text(-3.5, -4.5, s='Sepal length [standardized]', ha='center', va='center', fontsize=12)
    plt.text(-10.5, 4.5, s='Petal length [standardized]', ha='center', va='center', fontsize=12, rotation=90)
    plt.show()
    
    #tuning
    params = {'decisiontreeclassifier__max_depth' : [1,2], 'pipeline-1__clf__C' : [0.001, 0.1, 100.0]}
    grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    for params, mean_score, scores in grid.grid_scores_:
        print('%0.3f+/-%0.2f %r' %(mean_score, scores.std() / 2, params))
        
    print('Best parameters: %s' % grid.best_params_)
    
    print('Accuracy: %.2f' % grid.best_score_)

if __name__ == '__main__':
    combination_algorithms()