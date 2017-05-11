'''
Created on 2017. 5. 10.

@author: ko
'''
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS(object):
    '''
    Sequential Backward Selection
    
    Parameters
    ----------------------------------
    estimator : object
            estimate feature performance
    
    k_features : int
            return features' number
            
    scoring : float
            performance score
    
    test_size : float
            test sample size
    
    random_state : 0 or 1
            0 == random, 1 != random
    
    Attributes
    -------------------------------------
    indices_ : tuple
            
    
    subsets_ : list
    
    scores_ : list
    
    k_score_ : list
    '''

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_seize=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_seize
        self.random_state = random_state
        
    def fit(self, X, y):
        '''
        Parameters
        -----------------
         X : shape = [n_samples, n_features]
            Training Vectors
            n_samples is the number of samples
            n_features is the number of features
            
         y : shape = [n_samples]
            Target values. Return
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -=1
            
            self.scores_.append(scores[best])
            
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, X):
        '''
        Parameters
        -----------------
         X : shape = [n_samples, n_features]
            Training Vectors
            n_samples is the number of samples
            n_features is the number of features
        '''
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        
        return score