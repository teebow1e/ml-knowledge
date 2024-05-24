import pandas as pd
import regression_utility as regression_utility
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import joblib

def Train():
    df = pd.read_csv("../data/update_dataset.csv")

    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparameters = {
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'brute',
        'leaf_size': 10,
        'metric': 'manhattan'
    }

    model = KNeighborsRegressor(**hyperparameters)
    model.fit(X, y)

    print(f"Hyperparameters: {hyperparameters}")

    regression_utility.evaluate(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, '../model/retrain_knn_regrressor.pkl')
    print("TRAINING FINISHED")

if __name__ == "__main__":
    Train()