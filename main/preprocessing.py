# !/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Nov 29 15:10:13 2020

@author: ALEXIS
"""

# Un texte -> tokenize -> ner
# Peut-être le transformer en dataframe pour faciliter le travail
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd

nltk.download('stopwords')


def tok(doc, sw=False):
    result = []
    for line in doc:
        tokenized = nltk.word_tokenize(line)
        if sw:
            stop_words = set(stopwords.words('english'))
            l = [word for word in tokenized if word not in stop_words and word.isalpha()]
            result.append(l)
        else:
            l = [word for word in tokenized]
            result.append(l)
    return result


def bow(doc, vectorizer=None):
    doc = tok(doc, sw=True)
    porter = PorterStemmer()
    docs = []
    for d in doc:
        li = []
        for word in d:
            li.append(porter.stem(word))
        docs.append(" ".join(li))
    if vectorizer is None:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(docs)
    else:
        bow = vectorizer.transform(docs)
    bow = bow.toarray()
    bow = pd.DataFrame(bow)
    return bow, vectorizer
