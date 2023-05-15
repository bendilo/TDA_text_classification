import pandas as pd
import numpy as np
import nltk
import re
import contractions

################################
# Text Preprocessing Functions
################################


# A function that removes stopwords and expands contractions
# in one line of text string.
def remove_stopwords_line(line):
    # Remove contractions in the string (i.e. switching 'don't' to 'do not'.)
    line = ' '.join([contractions.fix(expanded_word) 
                     for expanded_word in line.split()])
    # Include words only if not in the list of stop_words.
    line = ' '.join([word for word in line.split() if word not in sw_nltk])
    return line




# A function that removes stopwords and expands contractions
# in document consisting of multiple lines.
def remove_stopwords_document(document):
    #Create a list consisting of each line in the document.
    lines = list(filter(None, document.splitlines()))
    return'\n'.join([remove_stopwords_line(line) for line in lines])



# Performs preprocessing steps on a dataset consisting of text documents (on column of name 'text')
def preprocess_df(dataframe):
    # Create a copy of the dataframe
    df = dataframe.copy()
    
    # Split text into sentences and make it such that there is one row per line
    df['text'] = df['text'].apply(lambda x: '\n'.join(nltk.sent_tokenize(x)))
    
    # Lowercase all letters
    df['text'] = df['text'].str.lower()
    
    # Expand contractions (don't --> do not)
    # and remove all stopwords from each text document
    df['text'] = df['text'].apply(remove_stopwords_document)
    
    #Remove punctuation, numbers and special characters
    df['text'] = df['text'].apply(lambda x: x.replace('-',' '))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^ \nA-Za-z.?!]+', '', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[.?!]+', '', x))
    
    #remove extra newlines and/or spaces
    df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub('\n+', '\n', x))
    df['text'] = df['text'].apply(lambda x: '\n'.join([line.strip() 
                                                       for line in x.splitlines()]))
    
    return df
    
    

# Load the lemmatizer from the nltk-library.
lemma = nltk.WordNetLemmatizer() 

# Define a function that performs lemmatization on a text document
def lemmatize(document):
    # Create a list consisting of each line in the document
    lines = document.splitlines()
    lemmatized_lines = []

    # perform lemmatization on each line in the document separately,
    # and rejoin them together
    for line in lines:
        lemmatized_lines.append(" ".join([lemma.lemmatize(item) 
                                         for item in line.split()]))
    return "\n".join(lemmatized_lines)
    
    
    
    
############################################
# Text Embedding functions
############################################

#TF-IDF embeddings
# Import the TF-IDF vectorizer from the sklearn-library
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_matrix(document):
    # Initialize the TF-IDF vectorizer.
    # Considers terms in a document to be both unigrams and bigrams.
    vectorizer = TfidfVectorizer(ngram_range=(1,2)) 
    
    # Split the document into lines
    lines = document.splitlines()
    
    # Produce the TF-IDF vector corresponding to each line
    # and stack them in a matrix
    matrix = vectorizer.fit_transform(lines).toarray()
    
    return matrix
    
    
#s-BERT embeddings
# Load a pre-trained sentence-BERT model from sentence_transformers
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def sbert_matrix(document):
    # Split the document into lines.
    lines = document.splitlines()
    
    # Generate an s-BERT embedding for each line 
    # and stack them in a matrix.
    matrix = sbert_model.encode(lines)
    
    return matrix



#GloVe embeddings
gloveFile = "glove.6B.50d.txt"

def loadGloveModel(gloveFile):
    print("Loading GloVe model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
        
    model = {}
    
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("GloVe model loaded successfully")
    return model

# Load the GloVe model from the given directory
glove = loadGloveModel(gloveFile)


#A function that returns the GloVe embedding of a given word if it exists, and a 50-dimensional zero vector if not.
def glove_vectorize(word):
    try: 
        vector = glove[word]
        return vector
    except KeyError:
        return np.zeros(50)


# A function that returns the GloVe embedding of a given document.
def glove_matrix(document):
    lines = document.splitlines()
    matrix = []
    
    for line in lines:
        if not line == "":
            words = nltk.word_tokenize(line)
            word_vectors = [glove_vectorize(word) for word in words]
            sentence_vector = sum(word_vectors) / len(word_vectors)
            
            # Only include the sentence-vector if it is non-zero
            if np.count_nonzero(sentence_vector) > 1:
                matrix.append(sentence_vector)
                
    return np.array(matrix)
    
    

#########################################
# Metric on Text Embeddings
#########################################
#Here we define a function that returns the angular distance matrix given a document embedding matrix
#where the $i$-th row in the matrix is the sentence vector of the $i$-th sentence in a document. 
#The resulting distance matrix is a $k \times k$ matrix in which the $[i, j]$ term of the matrix
#represents the distance between sentences $i$ and $j$ in the angular distance metric.


def angular_distance_matrix(A):
    if A.ndim == 1:
        return np.array([0])
    
    n_row, n_col = A.shape
    
    # The matrix where the (i,j)-th entry is the dot product
    #between row vector x_i and row vector x_j of A
    K = A @ A.transpose()
    
    # The array where the i-th entry is the 
    #norm of the i-th row vector of A
    normA = np.sqrt(np.diag(K))

    #smooth out by making zero-entries non-zero
    normA[normA == 0] = 1e-10
    
    # Tile the norms of A such that X_i * X_j is the matrix where the 
    #(i,j)-th entry is ||x_i||*||x_j||
    X_i = np.tile(normA, (n_row, 1))
    X_j = X_i.transpose()
    
    # The (i,j)-th entry is the cosine similarity of x_i and x_j
    cs = K/(X_i*X_j)
    
    # Avoid potential numerical issues when using the arccos function
    cs[cs>1] = 1
    cs[cs<0] = 0
    
    # Calculates the normalized angular distance matrix of A
    D = 2*np.arccos(cs)/np.pi
    
    #Ensure that the diagonal is zero
    for i in range(len(D[0])):
        D[i,i] = 0
    
    return D
    
    
    
    
    
###################################################
# TDA on Text
###################################################


# Here we implement Zhu's SIF and SIFTS algorithms,
# defined in the following article:
# X. Zhu, “Persistent homology: An introduction and a new text representation for natural language processing,”
# Proceedings of the 23rd International Joint Conference on Artificial Intelligence, 2013.
# Available: https: //www.ijcai.org/Proceedings/13/Papers/288.pdf.

import gudhi as gd

def SIF(D):
    # Create a Vietoris Rips filtration for a distance matrix D
    # with simplices of maximum dimension 2.
    VR_complex = gd.RipsComplex(distance_matrix=D)
    filtration = VR_complex.create_simplex_tree(max_dimension=2)
    
    #Calculate the persistent homology in dimensions 0 and 1
    persistence_intervals = filtration.persistence()
    
    return persistence_intervals

def SIFTS(D):
    # Create a copy of the distance matrix D
    D_SIFTS = D.copy()
    # Modify the new distance matrix by requiring D(x_i,x_{i+1}) = 0 
    for i in range(len(D_SIFTS[0])-1):
        D_SIFTS[i,i+1]=0
        D_SIFTS[i+1,i]=0
    
    # Create a Vietoris Rips filtration for a distance matrix D_SIFTS 
    #with simplices of maximum dimension 2.
    VR_complex = gd.RipsComplex(distance_matrix=D_SIFTS)
    filtration = VR_complex.create_simplex_tree(max_dimension=2)
    
    #Calculate the persistent homology in dimensions 0 and 1
    persistence_intervals = filtration.persistence()
    
    return persistence_intervals
    
    



# A function that restricts a list of persistence intervals (The output format of SIFTS and SIF)
# to a specific dimension, and converts these intervals to a format compatible with "PersistenceImager".

# In particular:
# Changes elements of form tuple(dimension, (birth, death))
# To a list of the form [birth,death] in a fixed dimension.

from persim import PersistenceImager

# Returns a list of the persistence intervals in a fixed dimension n
def intervals_in_dimension(n, persistence_intervals):
    lst = []
    for item in persistence_intervals:
        if item[0] == n:
            # Exclude potential intervals with infinite persistence in dim 0,
            # because they are incompatible with persistence images
            if item[1][1] != np.inf:
                lst.append(list(item[1]))
    return lst
    
    
    
#######################################
# Machine Learning Functions
#######################################
from sklearn.preprocessing import FunctionTransformer
from persim import PersistenceImager


# Returns the text embedding function of user specified choice
# Input: "tfidf", "sbert" or "glove"
def text_to_matrix(embedding):
    if embedding == "tfidf":
        return tfidf_matrix
    elif embedding == "sbert":
        return sbert_matrix
    else:
        return glove_matrix
        


def text_to_image_transformer(algorithm="tfidf", embedding="SIFTS", dimension=1, sigma=0.001, grid_size=100):
    '''
    Returns a function transformer compatible with machine learning pipelines in sklearn
    that transforms an array of preprocessed text documents 
    to an array of corresponding persistence images with user specified parameters
    for a given embedding method.
    
    Variables:
    algorithm: "SIFTS" or "SIF" #choose which TDA-text algorithm to use
    embedding: "tfidf", "glove" or "sbert" #which embedding method to use
    dimension: 0 or 1  #which dimension in the persistence diagram to transform to persistence image. Default: 1
    sigma: float value #which value for sigma in the normal distribution in persim
    grid_size: size of grid in the persistence image. default: 100 (yielding an image of resolution 100x100)
    '''

    def text_to_image(text):
        A = text_to_matrix(embedding)(text)
        D = angular_distance_matrix(A)
        
        if algorithm == "SIFTS":
            intervals = intervals_in_dimension(dimension,SIFTS(D))
        else:
            intervals = intervals_in_dimension(dimension,SIF(D))
        
        pimgr = PersistenceImager(pixel_size = 1/grid_size, 
                              birth_range = (0, 1), 
                              pers_range = (0, 1), 
                              kernel_params = {'sigma': sigma})
        imgs = pimgr.transform(intervals)
        imgs_array = imgs.flatten()
        return imgs_array
    
    def corpus_to_images(array):
        return np.array([text_to_image(item) for item in array])
    
    return FunctionTransformer(corpus_to_images)
