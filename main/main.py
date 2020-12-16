import pipeline

RESTAURANTS_TRAIN_PATH = "/data/Restaurants_Train.xml"
RESTAURANTS_TEST_PATH = "/data/Restaurants_Test_Gold.xml"
LAPTOP_TRAIN_PATH = "/data/Laptop_Train.xml"
LAPTOP_TEST_PATH = "/data/Laptop_Test_Gold.xml"

if __name__ == '__main__':
    # On va exécuter notre pipeline sur les différents jeu de données fournis au format xml
    pipeline.create_pipeline("RESTAURANTS", RESTAURANTS_TRAIN_PATH, RESTAURANTS_TEST_PATH)
    pipeline.create_pipeline("LAPTOP", LAPTOP_TRAIN_PATH, LAPTOP_TEST_PATH)
