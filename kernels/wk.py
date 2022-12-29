import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pickle

def wk(doc1, doc2):
    print("Starting WK, creating bag of words...\n")
    clean_docs = [doc1,doc2]
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = "english") 

    train_data_features = vectorizer.fit_transform(clean_docs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(train_data_features)
    tfidf = tfidf.toarray() 
    return np.dot(tfidf[0],tfidf[1]) #returns dot product of given 2 documents

def wkFeatVecs(trainDocs, testDocs):

    n_features_min = 2000
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(trainDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(train_data_features)
    tfidf = tfidf.toarray()
    nTrainDocs = len(tfidf)
    GmatTrain = np.ones((nTrainDocs,nTrainDocs))
            
    n_features_train = len(tfidf[0])
    
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    
    train_data_features = vectorizer.transform(testDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidfTest = transformer.fit_transform(train_data_features)
    tfidfTest = tfidfTest.toarray()
    nTestDocs = len(tfidfTest)
    GmatTest = np.ones((nTestDocs,nTrainDocs))
    
    return tfidf, tfidfTest