import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def initialize(random_state=2024):
    df = pd.read_csv("./data/update_dataset.csv")
    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["strength"].astype(int)  # Ensure this is integer for classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
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

def main():
    random_states = [42, 2024, 1234, 5678, 91011, 3721893, 1720412, 0, 7348192, 123, 5498320]  # List of random states to test
    results = []
    for state in random_states:
        print("[!] testing random state:", state)
        X_train, X_test, y_train, y_test, X, y = initialize(random_state=state)
        model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean', algorithm='auto', leaf_size=10)
        model.fit(X_train, y_train)
        results.append(evaluate(model, X_train, y_train, X_test, y_test))
    
    print(results)

    # Optionally print or analyze results across different runs here
    # This could include summarizing mean/std of accuracy, precision, recall, F1 across the different splits
    # Example: print("Average Accuracy:", np.mean([result['accuracy'] for result in results]))

if __name__ == "__main__":
    main()
