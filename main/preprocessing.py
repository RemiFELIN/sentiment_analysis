
#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Nov 29 15:10:13 2020

@author: ALEXIS
"""

# Un texte -> tokenize -> ner
# Peut-être le transformer en dataframe pour faciliter le travail

import nltk
from nltk.corpus import stopwords
from scikit_learn import preprocessing

def tok(doc, sw=False):
    tokenized = nltk.word_tokenize(doc)
    if sw:
        stop_words = set(stopwords.words('english'))
        doc = [word for word in tokenized if word not in stop_words]
        return doc
    else:
        doc = [word for word in tokenized]
        return doc


def pos_tag(doc):
    doc = nltk.pos_tag(doc)
    return doc



def label_encode(column):
    le = preprocessing.LabelEncoder()
    le.fit(column)
    le.transform(column)
    
    return column