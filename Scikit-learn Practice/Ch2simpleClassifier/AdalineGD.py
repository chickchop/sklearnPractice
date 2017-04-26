'''
Created on 2017. 4. 24.

@author: ko
'''
import numpy as np

class AdalineGD(object):
    '''
    Adaptive Linear Neuron classifier.
        
    Attribute
    --------------------------------
    w_ : 1d-array
        weights after fitting
    
    errors_ : list
            number of misclassification in every epoch
    '''

    def __init__(self, eta = 0.01, n_iter = 50):
        '''  
        Parameters
        --------------------------
        eta : float
            learning rate(between 0 ~ 1.0)
        
        n_iter : int
                passes over training dataset.
    
        '''
        self.eta = eta
        self.n_iter = n_iter
        
    #training data
    def fit(self, X, y):
        '''
        Parameter
        ------------
        X : array - like, shape = [n_sample, n_features]
            training vectors.
            n_sample is the number of sample
            n_feature is the number of sample's feature
            
        y : array-like, shape = [n_sample]
            target values.
            
        return
        ----------------
        self : object
        '''
        self.w_ = np.zeros(1+X.hape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.n_iter
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
        return self
    
    #calculate net input
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+ self.w_[0]
    
    #compute linear activation
    def activation(self, X):
        return self.net_input(X)
    
    #predict
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
        