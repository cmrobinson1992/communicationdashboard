import numpy as np
import os
from scipy import spatial

class Glove(object):
    def __init__(self):
        pass

    def load_glove_model(self, glove_file=os.path.join("data/glove.6B.50d.txt")):
        """
        Loads the glove model from the file.
        NOTE: This function is given and does not have to be implemented or changed.

        Args:
                glove_file: file path for the glove model.

        Return:
                glove_model: the embedding for each word in a dict format.
        """
        print("Loading Glove Model")
        glove_model = {}
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model

    def find_similar_word(self, model, emmbeddings):
        """
        Finds nearest similar word based on euclidean distance.
        NOTE: This function is given and does not have to be implemented or changed.

        Args:
                model: The glove model in a dict format.
                embeddings: The embeddings for which the nearest similar word needs to be found.

        Return:
                nearest: The nearest similar word for the given embeddings.
        """
        nearest = sorted(
            model.keys(),
            key=lambda word: spatial.distance.euclidean(model[word], emmbeddings),
        )
        return nearest

    def transform(self, model, data, dimension=50):
        """
        Transform the given data to a numpy array of glove features by taking a mean of all the embeddings for each word in a given sentence. 
        For Autograder purposes, use sentence.split() to split the sentence. 
        No other preprocessing other than converting words to lower case is necessary. 
        Any token which is not present in the glove model should be ignored.
        Set embedding to vector of zeros if none of the words in a sentence are found in the glove model.
  
        Args:
                model: The glove model in a dict format.
                data: (N,) The list of sentences to be transformed to embeddings.
                dimension: Dimension of the output embeddings.

        Return:
                transformed_features: (N,D) The extracted glove embeddings for each sentence with mean of all words in a sentence.

        Hint:
            You may find try/except block useful to deal with word that does not exist in the model
        """
        
 #       raise NotImplementedError
        transformed_features = []
        for sentence in data:
            glove_embeddings = []
            words = sentence.lower().split()
            for word in words:
                try:
                    glove_embeddings.append(model[word])
                except:
                    continue
            if glove_embeddings:
                mean_embedding = np.mean(glove_embeddings, axis=0)
            else:
                mean_embedding = np.zeros(dimension)
            transformed_features.append(mean_embedding)
        return np.array(transformed_features)
    