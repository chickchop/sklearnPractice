'''
Created on 2017. 5. 13.

@author: ko
'''
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_circles


def main():
    #make case 1 sample
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.show()
    
    #classify sample 1 case by pca
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((50, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((50, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2') 
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()
    
    #classify sample 1 case by kernel pca
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0, 0], np.zeros((50, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((50, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2') 
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.show()
    
    #make case 2 sample
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.show()
    
    #classify sample 2 case by pca
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2') 
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()
    
    #classify sample 2 case by kernel pca
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2') 
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.show()
    
    #make sample 3
    X, y = make_moons(n_samples=100, random_state=123)
    
    ##classify sample 3 by kernel pca
    alphas, lambdas = rbf_kernel_pca_alpha(X, gamma=15, n_components=1)
    
    #add new observation
    x_new = X[25]
    x_proj = alphas[25]
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    
    #plot
    plt.scatter(alphas[y==0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='remapped point X[25}', marker='x', s=500)
    plt.legend(scatterpoints=1)
    plt.show()

def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation
    
    Parameters
    -------------------------
    X : {Numpy ndarray}, shape = [n_samples, n_features]
    
    gamma : float
            Tuning parameter of the RBF kernel
            
    n_componets : int
                Number of principal components of return
                
    Returns
    -----------------------------
    X_pc : {Numpy ndarray}, shape = [n_samples, k_reatures]
        Projected dataset
    '''
    #Calculate pairwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    
    #Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    
    #compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    
    #center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(K)
    
    #collect the top k eigenvectors
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    
    return X_pc

def rbf_kernel_pca_alpha(X, gamma, n_components):
    '''
    RBF kernel PCA implementation
    
    Parameters
    -------------------------
    X : {Numpy ndarray}, shape = [n_samples, n_features]
    
    gamma : float
            Tuning parameter of the RBF kernel
            
    n_componets : int
                Number of principal components of return
                
    Returns
    -----------------------------
    alphas : {Numpy ndarray}, shape = [n_samples, k_reatures]
        Projected dataset
        
    lambdas : list
            eigenvalues
    '''
    #Calculate pairwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    
    #Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    
    #compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    
    #center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(K)
    
    #collect the top k eigenvectors
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1, n_components+1)))
    
    #collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    
    return alphas, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
        
    return k.dot(alphas / lambdas)


if __name__ == '__main__':
    main()