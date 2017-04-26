'''
Created on 2017. 4. 26.

@author: ko
'''
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from simpleClassifier import plot_decision_regions

def linear():
    #import data
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target
    
    #split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #feature scaling
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    #training
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    
    #plotting
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105,150))
    plt.xlabel('petal length[standardized')
    plt.ylabel('petal width[standardized')
    plt.legend(loc='upper left')
    plt.show()    
    
def xor():
    #XOR plotting
    np.random.seed(0)
    
    ##generate
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    ##ploting
    plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend()
    plt.show()
    
    #training
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()
    
def nonlinear():
    #import data
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target
    
    #split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    #feature scaling
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    #training
    svm = SVC(kernel='rbf', C=1.0, random_state=0, gamma=1)
    svm.fit(X_train_std, y_train)
   
    #plotting
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105,150))
    plt.xlabel('petal length[standardized')
    plt.ylabel('petal width[standardized')
    plt.legend(loc='upper left')
    plt.show()     
    
    
    
def plot_decision_regions(X, y, classifier, test_idx=None, resolution = 0.02):
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
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:, 1], s=55, c='', marker='o', alpha=1.0, linewidths=1, label='test set')    

        
if __name__ == '__main__':
    linear()
    xor()
    nonlinear()
    