'''
Created on 2017. 4. 19.

@author: ko
'''
import numpy as np


class Perceptron(object):
    '''
    Perceptron classifier.
    
    Parameters
    -----------------------------
    eta : float
        Learning rate(between 0 ~ 1.0)
        
    n_iter : int
        psddes over the training dataset.
        
    Attributes
    -----------------------------------
    w_ : 1d-array
        Weights after fitting.
        
    errors_ : list
        Number of misclassification in every epoch.
    '''
    def __init__(self, eta = 0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    #fit training data 
    def fit(self, X, y):
        '''
        parameters
        ----------------------
        X : shape = [n_samples, n_features]
            Training Vectors
            n_samples is the number of samples
            n_features is the number of features
            
        y : shape = [n_samples]
            Target values. Return
        '''
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
            
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1) 