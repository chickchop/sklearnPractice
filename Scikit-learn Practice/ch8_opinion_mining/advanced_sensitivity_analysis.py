'''
Created on 2017. 7. 7.

@author: ko
'''
import numpy as np
import re
import nltk
import pyprind

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


def tokenizer(text):
    '''
        Text data are transformed to original form of text token
        
        Parameters
        -----------------------
        text : String, shape = String data
            text data
            
        Returns
        -------------------------
        tokenized : array-list, shape = [n_text]
            text token and transformed to original form
    '''
    #import stopwords
    #nltk.download('stopwords')
    stop = stopwords.words('english')
    
    #delete tag
    text = re.sub('<[^>]*>', '', text)
    #find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    text = re.sub('[\W]+', ' ', text.lower())+' '.join(emoticons).replace('-', '')
    
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

def stream_docs(path):
    '''
        Generator
        read csv file and split text/label
        
        Parameter
        -----------------
        path : String
            file path
            
        Returns
        -------------------
        text : String
            text data
        label: String
            label data
    '''
    with open(path, 'r', encoding='UTF8') as csv:
        next(csv)
        
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
                                          
def get_minibatch(doc_stream, size):
    docs, y = [], []
    
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
            
    except StopIteration:
        return None, None
    
    return docs, y

def out_of_core_learning():
    vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    doc_stream = stream_docs(path='C:\\Users\\ko\\Desktop\\movie_data.csv')
    
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    
    #training model
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()
    
    #show accuracy    
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' %clf.score(X_test, y_test))
    
    #last update model
    clf = clf.partial_fit(X_test, y_test)

if __name__ == '__main__':
    out_of_core_learning()