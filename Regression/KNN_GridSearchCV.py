import regression_utility as regression_utility
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test, X, y = regression_utility.initialize()

hyperparameters = {
    'n_neighbors': [1, 3, 5, 10, 15, 20],
    'weights': ['distance'],
    'algorithm': ['brute'],
    'leaf_size': [10],
    'metric': ['manhattan']
}

model = KNeighborsRegressor()

grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=10, scoring='neg_mean_squared_error', verbose=3)

print("\n----- Training phase: -----")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n----- Best model evaluation: -----")
best_params = regression_utility.evaluate(best_model, X_train, y_train, X_test, y_test)

print("\n----- Best Hyperparameters: -----")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

print("\nBest score achieved:", grid_search.best_score_)

"""
----- Training phase: -----
Fitting 10 folds for each of 480 candidates, totalling 4800 fits

----- Best model evaluation: -----

----- Validation phase: -----
MAE (Validation): 0.0000
MSE (Validation): 0.0000
RMSE (Validation): 0.0000
R2 (Validation): 1.0000


----- Testing phase: -----
MAE (Testing): 0.0000
MSE (Testing): 0.0000
RMSE (Testing): 0.0002
R2 (Testing): 1.0000


----- Best Hyperparameters: -----
algorithm: brute
leaf_size: 10
n_neighbors: 7
weights: distance

Best score achieved: -5.037333308538695e-07

real    243m46.526s
user    348m56.290s
sys     315m46.032s
"""


"""
Second test:
MAE (Validation): 0.0000
MSE (Validation): 0.0000
RMSE (Validation): 0.0000
R2 (Validation): 1.0000


----- Testing phase: -----
MAE (Testing): 0.0000
MSE (Testing): 0.0000
RMSE (Testing): 0.0003
R2 (Testing): 1.0000


----- Best Hyperparameters: -----
algorithm: brute
leaf_size: 10
metric: manhattan
n_neighbors: 5
weights: distance

Best score achieved: -5.152920970993214e-07

real    556m4.431s
user    549m9.687s
sys     76m57.105s
"""