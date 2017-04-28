'''
Created on 2017. 4. 28.

@author: ko
'''
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.data import OneHotEncoder

def main1():
    #generate data
    csv_data = '''a,b,c,d
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0'''
    df = pd.read_csv(StringIO(csv_data))
    print(df)
    print(df.isnull().sum())
    
    #drop NaN
    '''df.dropna()'''
    
    #compensation data
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    print(imputed_data)
    
def main2():
    #generate data
    df = pd.DataFrame([['green', 'M', 10.1,'class1'],
                       ['red', 'L', 13.5, 'class2'],
                       ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']
    print(df)
    
    #make ordered label
    size_mapping = {'XL':3, 'L':2, 'M':1}
    df['size'] = df['size'].map(size_mapping)
    print(df)
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
    df['classlabel'] = df['classlabel'].map(class_mapping)
    print(df)
    inv_class_mapping = {v:k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print(df)
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    print(y)
    
    #one-hot encode
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:,0])
    print(X)
    ohe = OneHotEncoder(categorical_features=[0])
    X = ohe._fit_transform(X).toarray()
    print(X)
    print(pd.get_dummies(df[['price', 'color', 'size']]))
    
if __name__ == '__main__':
    main2()