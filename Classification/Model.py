import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def initialize():
    df = pd.read_csv("../data/outlier_included.csv")
    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"].astype(int)  # Ensure this is integer for classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
    return X_train, X_test, y_train, y_test, X, y

def evaluate(model, X_train, y_train, X_test, y_test):
    print("\n----- Validation phase: -----")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"Accuracy (Validation): {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Precision (Validation): {precision_score(y_train, y_pred_train, average='macro'):.4f}")
    print(f"Recall (Validation): {recall_score(y_train, y_pred_train, average='macro'):.4f}")
    print(f"F1 Score (Validation): {f1_score(y_train, y_pred_train, average='macro'):.4f}")

    print("\n----- Testing phase: -----")
    print(f"Accuracy (Testing): {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"Precision (Testing): {precision_score(y_test, y_pred_test, average='macro'):.4f}")
    print(f"Recall (Testing): {recall_score(y_test, y_pred_test, average='macro'):.4f}")
    print(f"F1 Score (Testing): {f1_score(y_test, y_pred_test, average='macro'):.4f}")

    return model.get_params()

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return result

def plot_permutation_importance(model, X, y, permutation_importance_result):
    feature_names = X.columns
    importances_mean = permutation_importance_result.importances_mean
    importances = pd.DataFrame(importances_mean, index=feature_names, columns=['Importance'])
    ax = importances.plot.barh(width = 0.8)
    ax.set_title("Permutation Importances")
    ax.set_xlabel("Mean importance score")
    plt.tight_layout()
    plt.show()
