'''
Created on 2017. 7. 7.

@author: ko
'''
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer


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
    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    #delete tag
    text = re.sub('<[^>]*>', '', text)
    #find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    text = re.sub('[\W]+', ' ', text.lower())+' '.join(emoticons).replace('-', '')
    
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

def stream_docs(path):
    with open(path, 'r') as csv:
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


if __name__ == '__main__':
    pass