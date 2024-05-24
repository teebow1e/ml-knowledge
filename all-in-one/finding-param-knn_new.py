from Initialize import *
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def find_regression():
    X_train, X_test, y_train, y_test, X, y = initialize_regression()
    hyperparameters = {
        'n_neighbors': [3, 5, 10, 50, 100, 500, 1000, 5000],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    model = KNeighborsRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=10, scoring='neg_mean_squared_error', verbose=1)

    print("\n----- Training phase: -----")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n----- Best model evaluation: -----")
    evaluate_regression(best_model, X_train, y_train, X_test, y_test)

    print("\n----- Best Hyperparameters: -----")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    print("\nBest score achieved:", grid_search.best_score_)

def find_classification():
    X_train, X_test, y_train, y_test, X, y = initialize_classification()
    hyperparameters = {
        'n_neighbors': [3, 5, 10, 50, 100, 500, 1000, 5000],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=10, scoring='accuracy', verbose=3)

    print("\n----- Training phase: -----")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n----- Best model evaluation: -----")
    evaluate_classification(best_model, X_train, y_train, X_test, y_test)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print("\nBest score achieved:", grid_search.best_score_)

if __name__ == "__main__":
    find_regression()
    find_classification()