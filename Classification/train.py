import pandas as pd
import numpy as np
import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

def Train():
    df = pd.read_csv("../data/outlier_included.csv")

    label_to_num = {
        'Very weak': 1,
        'Week': 2,
        'Average': 3,
        'Strong': 4,
        'Very strong': 5
    }

    # df['class_strength'] = df['class_strength'].map(label_to_num)
    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["class_strength"]


    # random_states = [0, 42, 100, 1000, 10000, 1337, 7331, 31337, 123456, 654321, 6321737, 23813385]
    # for rs in random_states:
    #     print(f"testinng random state {rs}")
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

    hyperparameters = {
        'n_neighbors': 3,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 10,
        'metric': 'euclidean'
    }

    model = KNeighborsClassifier(**hyperparameters)
    model.fit(X, y)

    print("Hyperparameters")
    print(hyperparameters)
    # Model.evaluate(model, X_train, y_train, X_test, y_test)

    print("TRAINING FINISHED")
    joblib.dump(model, '../model/final/knn_classification.pkl')

if __name__ == "__main__":
    Train()
