'''
Created on 2017. 7. 10.

@author: ko
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def hierarchical_tree_example():
    #random sample generate
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5,3])*10
    df = pd.DataFrame(X, columns=variables, index=labels)

    #calculate distance
    row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
    
    row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    
    #make dendrogram black
    row_dendr = dendrogram(row_clusters, labels=labels, color_threshold=np.inf)
    
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()

if __name__ == '__main__':
    hierarchical_tree_example()