from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from nltk.stem import PorterStemmer
import preprocessing
import dataset
import algo
import pickle

"""
Nous allons classifier les différents termes receuillis afin de construire notre modèle de prédiction
"""


def create_pipeline_v1(name, file_train, file_test):
    print("\n###############################################")
    print("# V1 - TRAIN " + name)
    print("###############################################\n")

    # On récupère notre dataset
    # restaurants = dataset.generate_dataset_from_file("/data/Restaurants_Train.xml")
    restaurants = dataset.generate_dataset_from_file(file_train)
    # On importe notre fichier de test tel que :
    # restaurants_test = dataset.generate_dataset_from_file("/data/Restaurants_Test_Gold.xml")
    restaurants_test = dataset.generate_dataset_from_file(file_test)

    # On construit nos variables
    X_train = []
    y_train = []
    for line in restaurants:
        # print(line[1])
        X_train.append(line[0] + " " + line[1])
        y_train.append(line[2])

    print("len(y_train):", len(y_train))
    print("list classes:", list(set(y_train)))

    print("\nlen(X_train):", len(X_train))
    # On va vectoriser notre dataset X_train tel que:
    bow = preprocessing.bow(X_train)
    X_train = bow[0]
    vectorizer = bow[1]
    print(X_train)

    # Utilisation de "RandomForest"
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("\n###############################################")
    print("# V1 - TEST " + name)
    print("###############################################\n")
    # On vient tester notre modèle avec le fichier de test
    # On construit nos variables
    X_test = []
    y_test = []
    for line in restaurants_test:
        X_test.append(line[0] + " " + line[1])
        y_test.append(line[2])

    print("y:", y_test)
    print("\nlen(y):", len(y_test))
    print("list classes:", list(set(y_test)))

    print("\nlen(X):", len(X_test))
    # On va ajouter notre dataset 'X_test' dans notre vectoriser:
    bow = preprocessing.bow(X_test, vectorizer)
    X_test = bow[0]
    print(X_test)

    # On test avec notre modèle
    y_pred_test = clf.predict(X_test)
    print("\nMODELE: test sur la variable X_test:"
          "\nAccuracy:", accuracy_score(y_test, y_pred_test),
          "\nMCC:", matthews_corrcoef(y_test, y_pred_test),
          "\nPrecision:", precision_score(y_test, y_pred_test, average='weighted'),
          "\nRecall:", recall_score(y_test, y_pred_test, average='weighted'),
          "\nF-measure:", f1_score(y_test, y_pred_test, average='weighted'))


def create_pipeline_v2(name, file_train, file_test):
    print("\n###############################################")
    print("# V2 - TRAIN " + name)
    print("###############################################\n")

    # On récupère notre dataset
    # restaurants = dataset.generate_dataset_from_file("/data/Restaurants_Train.xml")
    restaurants = dataset.generate_dataset_from_file(file_train)
    # On importe notre fichier de test tel que :
    # restaurants_test = dataset.generate_dataset_from_file("/data/Restaurants_Test_Gold.xml")
    restaurants_test = dataset.generate_dataset_from_file(file_test)

    # On construit nos variables
    X_train = []
    y_train = []
    for line in restaurants:
        if line[0] not in X_train:
            X_train.append(line[0])
            y_train.append(line[2])

    print("len(y_train):", len(y_train))
    print("list classes:", list(set(y_train)))

    print("\nlen(X_train):", len(X_train))
    # On va vectoriser notre dataset X_train tel que:
    bow = preprocessing.bow(X_train)
    X_train = bow[0]
    vectorizer = bow[1]
    print(X_train)

    # Utilisation de "RandomForest"
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("\n###############################################")
    print("# V2 - TEST " + name)
    print("###############################################\n")
    # On vient tester notre modèle avec le fichier de test
    # On construit nos variables
    X_test = []
    y_test = []
    for line in restaurants_test:
        X_test.append(line[0])
        y_test.append(line[2])

    print("y:", y_test)
    print("\nlen(y):", len(y_test))
    print("list classes:", list(set(y_test)))

    print("\nlen(X):", len(X_test))
    # On va ajouter notre dataset 'X_test' dans notre vectoriser:
    bow = preprocessing.bow(X_test, vectorizer)
    X_test = bow[0]
    print(X_test)

    # On test avec notre modèle
    y_pred_test = clf.predict(X_test)
    print("\nMODELE: test sur la variable X_test:"
          "\nAccuracy:", accuracy_score(y_test, y_pred_test),
          "\nMCC:", matthews_corrcoef(y_test, y_pred_test),
          "\nPrecision:", precision_score(y_test, y_pred_test, average='weighted'),
          "\nRecall:", recall_score(y_test, y_pred_test, average='weighted'),
          "\nF-measure:", f1_score(y_test, y_pred_test, average='weighted'))

