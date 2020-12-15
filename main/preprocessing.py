# Un texte -> tokenize -> ner
# Peut-être le transformer en dataframe pour faciliter le travail

import nltk
from nltk.corpus import stopwords
from scikit_learn import preprocessing
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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


def bow(doc,n):
    porter = PorterStemmer()
    doc=porter.stem(doc)
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(doc)
    bow = bow.toarray()
    bow= pd.DataFrame(bow)
    df_bow=pd.concat([df['target'],bow], axis=1)
    
    return df_bow


def tfidf(doc,n):
    porter = PorterStemmer()
    doc=porter.stem(doc)
    tfidf = TfidfVectorizer()
    tfidf_df = tfidf.fit_transform(doc)
    tfidf_df = tfidf_df.toarray()
    tfidf_df= pd.DataFrame(tfidf_df)
    df_tfidf=pd.concat([df['target'],tfidf_df], axis=1)
    
    return df_tfidf