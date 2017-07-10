'''
Created on 2017. 7. 10.

@author: ko
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def k_means_example():
    X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    
    plt.scatter(X[:,0], X[:,1], s=50, c='black', marker='o')
    plt.grid()
    plt.show()
    
    km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)
    
    #plot
    plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c='lightgreen', marker='s', label='cluster1')
    plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c='orange', marker='o', label='cluster2')
    plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c='lightblue', marker='v', label='cluster3')
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, c='red', marker='*', label='centroids')
    plt.legend()
    plt.grid()
    plt.show()
    
    print('Distortion: %.2f' %km.inertia_)
    
    #elbow method
    distortions = []
    for i in range(1, 11):
        km =KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
        
    plt.plot(range(1,11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

if __name__ == '__main__':
    k_means_example()