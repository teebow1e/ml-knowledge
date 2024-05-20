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
    y = df["strength"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test, X, y

def evaluate(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_precision': precision_score(y_train, y_pred_train, average='macro'),
        'test_precision': precision_score(y_test, y_pred_test, average='macro'),
        'train_recall': recall_score(y_train, y_pred_train, average='macro'),
        'test_recall': recall_score(y_test, y_pred_test, average='macro'),
        'train_f1': f1_score(y_train, y_pred_train, average='macro'),
        'test_f1': f1_score(y_test, y_pred_test, average='macro')
    }

    print("\n----- Validation phase: -----")
    print(f"Accuracy (Validation): {results['train_accuracy']:.4f}")
    print(f"Precision (Validation): {results['train_precision']:.4f}")
    print(f"Recall (Validation): {results['train_recall']:.4f}")
    print(f"F1 Score (Validation): {results['train_f1']:.4f}")

    print("\n----- Testing phase: -----")
    print(f"Accuracy (Testing): {results['test_accuracy']:.4f}")
    print(f"Precision (Testing): {results['test_precision']:.4f}")
    print(f"Recall (Testing): {results['test_recall']:.4f}")
    print(f"F1 Score (Testing): {results['test_f1']:.4f}")

    return results

def main():
    random_states = [0, 42, 100, 1337, 2024, 1234, 5678, 7331, 91011, 758290, 3721893, 1720412, 7348192, 5498320, 71823934, 4294967294]
    results = []
    for state in random_states:
        print("[!] testing random state:", state)
        X_train, X_test, y_train, y_test, X, y = initialize(random_state=state)
        model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean', algorithm='auto', leaf_size=10)
        model.fit(X_train, y_train)
        results.append(evaluate(model, X_train, y_train, X_test, y_test))
        print()
    
    avg_results = {
        'avg_train_accuracy': np.mean([result['train_accuracy'] for result in results]),
        'avg_test_accuracy': np.mean([result['test_accuracy'] for result in results]),
        'avg_train_precision': np.mean([result['train_precision'] for result in results]),
        'avg_test_precision': np.mean([result['test_precision'] for result in results]),
        'avg_train_recall': np.mean([result['train_recall'] for result in results]),
        'avg_test_recall': np.mean([result['test_recall'] for result in results]),
        'avg_train_f1': np.mean([result['train_f1'] for result in results]),
        'avg_test_f1': np.mean([result['test_f1'] for result in results])
    }

    print("\nAverage Results:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

"""
[!] testing random state: 0

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

[!] testing random state: 42

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

[!] testing random state: 100

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

[!] testing random state: 1337

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

[!] testing random state: 2024

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

[!] testing random state: 1234

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

[!] testing random state: 5678

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

[!] testing random state: 7331

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

[!] testing random state: 91011

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

[!] testing random state: 758290

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

[!] testing random state: 3721893

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

[!] testing random state: 1720412

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

[!] testing random state: 7348192

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

[!] testing random state: 5498320

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

[!] testing random state: 71823934

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

[!] testing random state: 4294967294

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


Average Results:
avg_train_accuracy: 1.0000
avg_test_accuracy: 1.0000
avg_train_precision: 1.0000
avg_test_precision: 1.0000
avg_train_recall: 1.0000
avg_test_recall: 1.0000
avg_train_f1: 1.0000
avg_test_f1: 1.0000
"""