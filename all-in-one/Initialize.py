import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import csv

def initialize_regression(RandomState):
    df = pd.read_csv("data/update_dataset2.csv")

    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RandomState)

    return X_train, X_test, y_train, y_test, X, y

def initialize_classification(RandomState):
    df = pd.read_csv("data/update_dataset2.csv")

    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["class_strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RandomState)

    return X_train, X_test, y_train, y_test, X, y

def plot(model, X, y, permutation_importance_result):
    feature_names = X.columns

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))


    axes.barh(feature_names, model.feature_importances_)
    axes.set_xlabel('Feature Importance (MDI)')
    axes.set_ylabel('Feature')
    axes.set_title('Feature Importance Scores (MDI)')

    # # Plot MDI feature importance
    # axes[0].barh(feature_names, model.feature_importances_)
    # axes[0].set_xlabel('Feature Importance (MDI)')
    # axes[0].set_ylabel('Feature')
    # axes[0].set_title('Feature Importance Scores (MDI)')

    # # Plot permutation importance 
    # importances_mean = permutation_importance_result.importances_mean
    # importances = pd.DataFrame(importances_mean, index=feature_names, columns=['Importance'])
    # ax = importances.plot.barh(ax=axes[1], width = 0.8)
    # ax.set_title("Permutation Importances")
    # ax.set_xlabel("Mean importance score")

    plt.tight_layout()
    plt.show()

def train_regression(model, X_train, y_train, hyperparameters):
    clf = GridSearchCV(model, hyperparameters, cv=10, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    # print("Best parameters found:", clf.best_params_)
    return clf.best_estimator_

def train_classification(model, X_train, y_train, hyperparameters):
    clf = GridSearchCV(model, hyperparameters, cv=10, scoring='accuracy')
    clf.fit(X_train, y_train)

    # print("Best parameters found for classification:", clf.best_params_)
    return clf.best_estimator_

def evaluate_regression(model, X_train, y_train, X_test, y_test):
    # print("\n----- Validation phase: -----")
    y_pred_train = model.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    # print(
    #     f"MAE (Validation): {mae_train:.4f}\n"
    #     f"MSE (Validation): {mse_train:.4f}\n"
    #     f"RMSE (Validation): {rmse_train:.4f}\n"
    #     f"R2 (Validation): {r2_train:.4f}\n"
    # )

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

def evaluate_classification(model, X_train, y_train, X_test, y_test):
    print("\n----- Validation phase (Classification): -----")
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Training Accuracy:", accuracy_train)
    print("Classification Report (Training):")
    print(classification_report(y_train, y_pred_train, digits=4))

    print("\n----- Testing phase (Classification): -----")
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Testing Accuracy:", accuracy_test)
    print("Classification Report (Testing):")
    print(classification_report(y_test, y_pred_test, digits=4))
    print_confusion_matrix(y_test, y_pred_test, "Testing")
    return model.get_params() 

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return result

def print_confusion_matrix(y_true, y_pred, phase):
    print(f"Confusion Matrix ({phase}):")
    print(confusion_matrix(y_true, y_pred))

