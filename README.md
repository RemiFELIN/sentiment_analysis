# sentiment_analysis

En gardant une procédure restreinte, nous arrivons (étonnamment) à 100% de réussite sur le jeu de données de test de restaurant.  

Notre préprocessing est le suivant :  
On utilise la tokénization en premier lieu.  
Ensuite, nous enlevons les stopword de la phrase (les mots 'inutiles', ou 'vide de sens').  
Maintenant, nous allons utiliser le Porter Stemmer, afin de retrouver les racines des mots, ce qui va nous servir pour la suite.  
Enfin nous appliquons un Bag of word pour avoir un dataset avec en ligne : les phrases, et en colonnes : tous les mots présents dans notre xml.  

Le petit ajout, qui nous apporte un gain dans les métriques est l'ajout de l'aspect term à la fin de la phrase.   

Ensuite nous pouvons appliquer notre algorithme de machine learning pour prédire la polarité des aspect terms.  
Notre algorithme est Random Forest, avec les paramètres initiaux.  
  
Avec cette pipeline:  
- Nous arrivons à 100% de toutes les métriques dans le fichier test de restaurant.  
- Nous arrivons à 97.9% en Recall et F-measure, et 98% en Precision.  

Nous affichons aussi l'Accuracy et le MCC (matthews correlation coefficient).  
