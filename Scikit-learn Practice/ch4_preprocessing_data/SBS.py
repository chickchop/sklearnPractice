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
    
    k_scores :
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        