#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:10:13 2020

@author: ALEXIS
"""


#Un texte -> tokenize -> ner 
#Peut-être le transformer en dataframe pour faciliter le travail
import spacy
import nltk
from nltk.corpus import stopwords  




def tok(doc, sw=False):
    tokenized = nltk.word_tokenize(doc)
    if sw == True:
        stop_words = set(stopwords.words('english'))  
        doc = [word for word in tokenized if not word in stop_words]  
        return doc
    else:
        doc = [word for word in tokenized]
        return doc


