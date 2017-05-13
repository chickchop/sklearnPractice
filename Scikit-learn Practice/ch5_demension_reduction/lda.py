'''
Created on 2017. 5. 13.

@author: ko
'''
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

def make_lda():
    #import data
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    #split data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #standardization
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    
    #calculate mean vector
    np.set_printoptions(precision=4)
    mean_vecs = []
    
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
    #scaling discrete matrix
    d = 13 #number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    
    #calculate discrete matrix
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13
    S_B = np.zeros((d, d))
    
    for i, mean_vec in enumerate(mean_vecs):
        n = X[y==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    
    print('Between-class scatter matrix:%sx%s' % (S_B.shape[0], S_B.shape[1]))
    
    #make covariance
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    #sort covariance
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    
    print('Eigenvalues in descreasing order: \n')
    for eigen_vals in eigen_pairs:
        print(eigen_vals[0])
        
    #plot
    tot = sum(eigen_vals)
    discr = [(i / tot) for i in sorted(eigen_vals, reverse = True)]
    cum_discr = np.cumsum(discr)
    plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
    plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability')
    plt.ylabel('"discriminability" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.show()
    
def lda_sklearn():
    #import data
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    #split data
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #standardization
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
        
    #lda and classify by logistic regression
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    
    #plot
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend(loc='lower left')
    plt.show() 
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
            
if __name__ == '__main__':
    lda_sklearn()