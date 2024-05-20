import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the data
X_train, X_test, y_train, y_test, X, y = Model.initialize()

hyperparameters = {
    'n_neighbors': [3, 5, 10, 50, 100, 500, 1000, 5000],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

model = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=10, scoring='accuracy', verbose=1)

print("\n----- Training phase: -----")
# Perform grid search
grid_search.fit(X_train, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

print("\n----- Best model evaluation: -----")
# Evaluate the best model using your Model module's evaluate method
best_params = Model.evaluate(best_model, X_train, y_train, X_test, y_test)

print("\n----- Best Hyperparameters: -----")
# Print the best hyperparameters
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# Optionally, print best score achieved
print("\nBest score achieved:", grid_search.best_score_)


"""
First try:
----- Training phase: -----
Fitting 10 folds for each of 480 candidates, totalling 4800 fits

----- Best model evaluation: -----

----- Validation phase: -----
Accuracy (Validation): 1.0000
Precision (Validation): 1.0000
Recall (Validation): 1.0000
F1 Score (Validation): 1.0000

----- Testing phase: -----
Accuracy (Testing): 1.0000
Precision (Testing): 1.0000
Recall (Testing): 1.0000
F1 Score (Testing): 1.0000

----- Best Hyperparameters: -----
algorithm: auto
leaf_size: 10
n_neighbors: 3
weights: uniform

Best score achieved: 1.0

real    271m13.357s
user    351m57.180s
sys     318m10.819s
"""

"""
----- Training phase: -----
Fitting 10 folds for each of 192 candidates, totalling 1920 fits

----- Best model evaluation: -----

----- Validation phase: -----
Accuracy (Validation): 1.0000
Precision (Validation): 1.0000
Recall (Validation): 1.0000
F1 Score (Validation): 1.0000

----- Testing phase: -----
Accuracy (Testing): 1.0000
Precision (Testing): 1.0000
Recall (Testing): 1.0000
F1 Score (Testing): 1.0000

----- Best Hyperparameters: -----
algorithm: auto
leaf_size: 10
n_neighbors: 3
weights: uniform

Best score achieved: 1.0

real    140m30.548s
user    142m58.822s
sys     31m4.621s
"""

"""
'n_neighbors': [3, 30, 100, 300, 1000, 3000, 10000],

----- Training phase: -----
Fitting 10 folds for each of 224 candidates, totalling 2240 fits

----- Best model evaluation: -----

----- Validation phase: -----
Accuracy (Validation): 1.0000
Precision (Validation): 1.0000
Recall (Validation): 1.0000
F1 Score (Validation): 1.0000

----- Testing phase: -----
Accuracy (Testing): 1.0000
Precision (Testing): 1.0000
Recall (Testing): 1.0000
F1 Score (Testing): 1.0000

----- Best Hyperparameters: -----
algorithm: auto
leaf_size: 10
n_neighbors: 3
weights: uniform

Best score achieved: 1.0

real    683m12.228s
user    672m21.107s
sys     41m15.026s
"""

"""
Try with more params:
----- Best Hyperparameters: -----
algorithm: auto
leaf_size: 10
metric: euclidean
n_neighbors: 3
weights: uniform
"""