from Initialize import *
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def Regression():
    X_train, X_test, y_train, y_test, X, y = initialize_regression()
    model = KNeighborsRegressor()
    hyperparameters = {
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'brute',
        'leaf_size': 10,
        'metric': 'manhattan'
    }

    best_model = train_regression(model, X_train, y_train, hyperparameters)

    print("\n----- Best model evaluation: -----")
    best_params = evaluate_regression(best_model, X_train, y_train, X_test, y_test)

    print("\n----- Overall Parameter use: -----")
    for param, value in best_params.items():
        print(f"{param}: {value}")


def Classification():
    X_train, X_test, y_train, y_test, X, y = initialize_classification()
    model = KNeighborsClassifier()
    hyperparameters = {
        'n_neighbors': 3,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 10,
        'metric': 'euclidean'
    }

    best_model = train_classification(model, X_train, y_train, hyperparameters)

    print("\n----- Best model evaluation: -----")
    best_params = evaluate_classification(best_model, X_train, y_train, X_test, y_test)

    print("\n----- Overall Parameter use: -----")
    for param, value in best_params.items():
        print(f"{param}: {value}")


if __name__ == "__main__":
    Classification()
    Regression()