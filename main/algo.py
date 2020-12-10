

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
#import xgboost as xgb
#from sklearn.ensemble import GradientBoostingClassifier


"""
Nous allons procéder suivant l'état de l'art du machine learning
Nous n'allons pas nous contenter de l'accuracy, car c'est une métrique biaisée
dans les cas des datasets unbalanced (accuracy paradox).
Nous allons donc aussi utiliser le MCC, qui va prendre en compte toutes les
facettes de la matrice de confusion.

"""
def random_forest(df, n_estimators):
    
    y = df.target
    X = df.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    rf=RandomForestClassifier(n_estimators=n_estimators)

    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred),"MCC:",metrics.matthews_corrcoef(y_test, y_pred))
    
    return rf


def knn(df,n_neighbors):
    y = df.target
    X = df.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred),"MCC:",metrics.matthews_corrcoef(y_test, y_pred) )

    return knn



