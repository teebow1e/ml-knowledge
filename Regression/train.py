import pandas as pd
import Model
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

def Train():
    df = pd.read_csv("../data/outlier_included.csv")

    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"]



    hyperparameters = {
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'brute',
        'leaf_size': 10,
        'metric': 'manhattan'
    }

    model = KNeighborsRegressor(**hyperparameters)
    model.fit(X, y)

    print("Hyperparameters")
    print(hyperparameters)
        
    
    joblib.dump(model, '../model/final/knn_regressor.pkl')
    print("done")

if __name__ == "__main__":
    Train()
