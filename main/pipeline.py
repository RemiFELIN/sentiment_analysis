from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from nltk.stem import PorterStemmer
import preprocessing
import dataset
import algo
import pickle

"""
Nous allons classifier les différents termes receuillis afin de construire notre modèle de prédiction
"""


def create_pipeline(name, file_train, file_test):

    print("\n###############################################")
    print("# TRAIN " + name)
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
        X_train.append(line[0] + " " + line[1])
        y_train.append(line[2])

    print("\nlen(y_train):", len(y_train))
    print("list classes:", list(set(y_train)))
    nb_classes = len(list(set(y_train)))
    # On encode y tel que :
    # y_train = preprocessing.label_encode(y_train)

    print("\nlen(X_train):", len(X_train))
    # On va vectoriser notre dataset X_train tel que:
    bow = preprocessing.bow(X_train)
    X_train = bow[0]
    vectorizer = bow[1]
    print(X_train)

    # Utilisation de "RandomForest"
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print("MODELE: test sur la variable X_train:\n"
          "Accuracy:", accuracy_score(y_train, y_pred), "\nMCC:", matthews_corrcoef(y_train, y_pred))

    print("\n###############################################")
    print("# TEST " + name)
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
    nb_classes = len(list(set(y_test)))

    print("\nlen(X):", len(X_test))
    # On va ajouter notre dataset 'X_test' dans notre vectoriser:
    X_test = preprocessing.bow(X_test, vectorizer)[0]
    print("X_test:\n", X_test)

    # On test avec notre modèle
    y_pred_test = clf.predict(X_test)
    print("MODELE: test sur la variable X_test:\n"
          "Accuracy:", accuracy_score(y_test, y_pred_test), "\nMCC:", matthews_corrcoef(y_test, y_pred_test))
