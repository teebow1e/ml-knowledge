from Initialize import *
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import joblib

def TrainRG():
    df = pd.read_csv("data/update_dataset.csv")
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

    evaluate_regression(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, 'knn_regressor.pkl')
    print("TRAINING FINISHED")


def TrainCL():
    df = pd.read_csv("data/update_dataset.csv")
    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["class_strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparameters = {
        'n_neighbors': 3,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 10,
        'metric': 'euclidean'
    }

    model = KNeighborsClassifier(**hyperparameters)
    model.fit(X, y)

    print(f"Hyperparameters: {hyperparameters}")

    evaluate_classification(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, 'knn_classification.pkl')
    print("TRAINING FINISHED")


if __name__ == "__main__":
    TrainCL()
    TrainRG()