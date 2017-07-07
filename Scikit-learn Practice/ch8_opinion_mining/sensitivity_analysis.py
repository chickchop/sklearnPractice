'''
Created on 2017. 7. 6.

@author: ko
'''
import pyprind
import pandas as pd
import os
import numpy as np
import re
import nltk

from nltk.stem.porter import PorterStemmer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def sentiment_analysis_data_import():
    #data import
    pbar = pyprind.ProgBar(50000)
    labels = {'pos' : 1, 'neg' : 0}
    df = pd.DataFrame()
    
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = 'C:\\Users\\ko\\Desktop\\aclImdb\\%s\\%s' %(s, l)
            
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='UTF8') as infile:
                    txt = infile.read()
                    
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
                
    df.columns = ['review', 'sentiment']
    
    #shuffling
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    
    #data save(csv file)
    df.to_csv('C:\\Users\\ko\\Desktop\\movie_data.csv', index=False, encoding='UTF8')
    
    
    
def text_preprocessor(text):
    #delete tag
    text = re.sub('<[^>]*>', '', text)
    #find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    text = re.sub('[\W]+', ' ', text.lower())+' '.join(emoticons).replace('-', '') 
    
    return text

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    '''
        Token text data are transformed to original form
        
        Parameters
        -----------------------
        text : String, shape = String data
            text data
            
        Returns
        -------------------------
        [porter.stem(word) for word in text.split()] : array-list, shape = [n_text]
            text token and transformed to original form
    '''
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]



def sentiment_analysis_example():
    #data import
    df = pd.read_csv('C:\\Users\\ko\\Desktop\\movie_data.csv')
    
    #data split(train/test)
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values
    
    #import stop word
    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    #text data vectorize
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    
    #train model
    param_grid = [{'vect__ngram_range' : [(1,1)],
                   'vect__stop_words' : [stop, None],
                   'vect__tokenizer' : [tokenizer, tokenizer_porter],
                   'clf__penalty' : ['l1', 'l2'],
                   'clf__C' : [1.0, 10.0, 100.0]},
                  {'vect__ngram_range' : [(1,1)],
                   'vect__stop_words' : [stop, None],
                   'vect__tokenizer' : [tokenizer, tokenizer_porter],
                   'vect__use_idf' : [False],
                   'vect__norm' : [None],
                   'clf__penalty' : ['l1', 'l2'],
                   'clf__C' : [1.0, 10.0, 100.0]}]
    
    ##make pipeline(logistic regression)
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])
    ##find best combination of parameter
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    
    #fitting
    gs_lr_tfidf.fit(X_train, y_train)
    
    #show best combination of parameter
    print('Best parameter set: %s' %gs_lr_tfidf.best_params_)
    
    #show model accuracy
    print('CV Accuracy : %.3f' %gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy : %.3f' %clf.score(X_test, y_test))
            
if __name__ == '__main__':
    #sentiment_analysis_data_import()
    #sentiment_analysis_example()