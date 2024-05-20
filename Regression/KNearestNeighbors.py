import Model
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()
X_train, X_test, y_train, y_test, X, y = Model.initialize()

hyperparameters = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40],
}

print("\n----- Training phase: -----")
best_model = Model.train(model, X_train, y_train, hyperparameters)
print("\n----- Best model evaluation: -----")
best_params = Model.evaluate(best_model, X_train, y_train, X_test, y_test)

print("\n----- Overall Parameter use: -----")
for param, value in best_params.items():
    print(f"{param}: {value}")

# permutation_importance_result = Model.calculate_permutation_importance(best_model, X_test, y_test)
# Model.plot(best_model, X, y, permutation_importance_result)

"""
----- Training phase: -----
Best parameters found: {'algorithm': 'brute', 'leaf_size': 10, 'n_neighbors': 7, 'weights': 'distance'}

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


----- Overall Parameter use: -----
algorithm: brute
leaf_size: 10
metric: minkowski
metric_params: None
n_jobs: None
n_neighbors: 7
p: 2
weights: distance
"""