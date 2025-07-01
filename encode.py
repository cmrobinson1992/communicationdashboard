import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))

class BagOfWords(object):
    def __init__(self):
        '''
        Initialize instance of CountVectorizer in self.vectorizer for use in fit and transform
        '''
        self.vectorizer = CountVectorizer()
  #      raise NotImplementedError

    def fit(self, data):
        '''
        Use the initialized instance of CountVectorizer to fit to the given data

        Args:
            data: list of N strings 
        
        Return:
            None
        '''
        self.vectorizer.fit(data)

   #     raise NotImplementedError

    def transform(self, data):
        '''
        Use the initialized instance of CountVectorizer to transform the given data
            
        Args:
            data: list of N strings
        
        Return:
            x: (N, D) bag of words numpy array

        Hint: .toarray() may be helpful
        '''
        return self.vectorizer.transform(data).toarray()
    #    raise NotImplementedError


class TfIdf(object):
    def __init__(self):
        '''
        Initialize instance of TfidfVectorizer in self.vectorizer for use in fit and transform
        '''
        self.vectorizer = TfidfVectorizer()
    #    raise NotImplementedError


    def fit(self, data):
        '''
        Use the initialized instance of TfidfVectorizer to fit to the given data

        Args:
            data: list of N strings 
        
        Return:
            None
        '''
        self.vectorizer.fit(data)
   #     raise NotImplementedError


    def transform(self, data):
        '''
        Use the initialized instance of TfidfVectorizer to transform the given data
            
        Args:
            data: list of N strings
        
        Return:
            x: (N, D) tfi-df numpy array

        Hint: .toarray() may be helpful
        '''
        return self.vectorizer.transform(data).toarray()
    #    raise NotImplementedError
# In encode.py (or a new module like glove.py)

class GloveEmbedding:
    def __init__(self, glove_path=os.path.join(script_dir, "Data\\glove.6B.50d.txt")):
        self.embeddings = {}
        self.dim = 100  # Change if you use a different GloVe file
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = vector

    def transform(self, documents):
        return np.array([self._document_vector(doc) for doc in documents])

    def _document_vector(self, doc):
        words = doc.split()
        valid_vectors = [self.embeddings[word] for word in words if word in self.embeddings]
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        else:
            return np.zeros(self.dim)
