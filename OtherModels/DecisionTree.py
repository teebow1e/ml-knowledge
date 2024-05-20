import Model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test, X, y = Model.initialize()

hyperparameters = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [1000, 2000, 3000, 4000],
    'min_samples_leaf': [100, 200, 500, 1000, 1500, 2000, 5000],
    'max_features': [1, 2, 4, 6, 8]
}

best_model = Model.train(model, X_train, y_train, hyperparameters)
print("\n----- Best model evaluation: -----")
best_params = Model.evaluate(best_model, X_train, y_train, X_test, y_test)

print("\n----- Overall Parameter use: -----")
for param, value in best_params.items():
    print(f"{param}: {value}")

permutation_importance_result = Model.calculate_permutation_importance(best_model, X_test, y_test)
Model.plot(best_model, X, y, permutation_importance_result)
