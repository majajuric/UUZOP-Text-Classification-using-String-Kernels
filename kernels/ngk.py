import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def ngk(doc1, doc2, n=2, mode='char', norm=True):
    # Counts the occurences of unique n-grams
    ngrams = CountVectorizer(analyzer=mode, ngram_range=(n, n)).fit_transform([doc1, doc2])

    # Optionally fit a Tfidf transform
    if norm:
        a, b = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        a, b = ngrams.toarray()

    return np.dot(a, b)

def ngkGmats(trainDocs,testDocs, n=2, mode='char', norm=True):
    vectorizer = CountVectorizer(analyzer=mode, ngram_range=(n, n))
    ngrams = vectorizer.fit_transform(trainDocs)
    
    if norm:
        featVecsTrain = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        featVecsTrain = ngrams.toarray()
        
    ngrams = vectorizer.transform(testDocs)
    
    if norm:
        featVecsTest = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        featVecsTest = ngrams.toarray()
        
    nTrainDocs = len(featVecsTrain)
    nTestDocs = len(featVecsTest)
    
    ngkTrainKmat = np.ones((nTrainDocs,nTrainDocs))

    for i in range( 0, nTrainDocs ):
        for j in range(0,nTrainDocs):
            ngkTrainKmat[i][j] = np.dot(featVecsTrain[i], featVecsTrain[j])
    
    ngkTestKmat = np.ones((nTestDocs,nTrainDocs))

    for i in range( 0, nTestDocs ):
        for j in range(0,nTrainDocs):
            ngkTestKmat[i][j] = np.dot(featVecsTest[i], featVecsTrain[j])
    
    return ngkTrainKmat, ngkTestKmat
