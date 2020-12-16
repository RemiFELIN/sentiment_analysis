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
    tokenized = nltk.word_tokenize(doc)
    if sw:
        stop_words = set(stopwords.words('english'))
        doc = [word for word in tokenized if word not in stop_words and word.isalpha()]
        return doc
    else:
        doc = [word for word in tokenized]
        return doc


def pos_tag(doc):
    doc = nltk.pos_tag(doc)
    return doc


def label_encode(y):
    le = preprocessing.LabelEncoder().fit(y)
    return le.transform(y)


def bow(doc, vectorizer=None):
    porter = PorterStemmer()
    docs = []
    for d in doc:
        docs.append(porter.stem(d))
    if vectorizer is None:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(docs)
    else:
        bow = vectorizer.transform(docs)
    bow = bow.toarray()
    bow = pd.DataFrame(bow)
    return bow, vectorizer


def tfidf(doc):
    porter = PorterStemmer()
    doc = porter.stem(doc)
    tfidf = TfidfVectorizer()
    tfidf_df = tfidf.fit_transform(doc)
    tfidf_df = tfidf_df.toarray()
    tfidf_df = pd.DataFrame(tfidf_df)
    df_tfidf = pd.concat([df['target'], tfidf_df], axis=1)
    return df_tfidf
