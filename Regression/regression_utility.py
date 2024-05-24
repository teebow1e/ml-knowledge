import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def initialize():
    df = pd.read_csv("../data/update_dataset.csv")

    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

    return X_train, X_test, y_train, y_test, X, y

def plot(model, X, y, permutation_importance_result):
    feature_names = X.columns

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    axes[0].barh(feature_names, model.feature_importances_)
    axes[0].set_xlabel('Feature Importance (MDI)')
    axes[0].set_ylabel('Feature')
    axes[0].set_title('Feature Importance Scores (MDI)')

    importances_mean = permutation_importance_result.importances_mean
    importances = pd.DataFrame(importances_mean, index=feature_names, columns=['Importance'])
    ax = importances.plot.barh(ax=axes[1], width = 0.8)
    ax.set_title("Permutation Importances")
    ax.set_xlabel("Mean importance score")

    plt.tight_layout()
    plt.show()

def train(model, X_train, y_train, hyperparameters):
    clf = GridSearchCV(model, hyperparameters, cv=4, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    print("Best parameters found:", clf.best_params_)
    return clf.best_estimator_

def evaluate(model, X_train, y_train, X_test, y_test):
    print("\n----- Validation phase: -----")
    y_pred_train = model.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    print(
        f"MAE (Validation): {mae_train:.4f}\n"
        f"MSE (Validation): {mse_train:.4f}\n"
        f"RMSE (Validation): {rmse_train:.4f}\n"
        f"R2 (Validation): {r2_train:.4f}\n"
    )

    print("\n----- Testing phase: -----")
    y_pred_test = model.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(
        f"MAE (Testing): {mae_test:.4f}\n"
        f"MSE (Testing): {mse_test:.4f}\n"
        f"RMSE (Testing): {rmse_test:.4f}\n"
        f"R2 (Testing): {r2_test:.4f}\n"
    )

    return model.get_params()  

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return result