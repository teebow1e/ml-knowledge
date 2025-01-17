import pandas as pd
import classification_utility
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

def Train():
    df = pd.read_csv("../data/update_dataset.csv")

    label_to_num = {
        'Very weak': 1,
        'Week': 2,
        'Average': 3,
        'Strong': 4,
        'Very strong': 5
    }

    # df['class_strength'] = df['class_strength'].map(label_to_num)
    X = df.drop(["password", "strength", "class_strength"], axis=1)
    y = df["class_strength"]

    print(y.head())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparameters = {
        'n_neighbors': 3,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 10,
        'metric': 'euclidean'
    }

    print(f"sklearn version: {sklearn.__version__}")
    print(f"joblib version: {joblib.__version__}")
    model = KNeighborsClassifier(**hyperparameters)
    model.fit(X, y)

    print(f"Hyperparameters: {hyperparameters}")
    
    classification_utility.evaluate(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, '../model/retrain_knn_classification.pkl')
    print("TRAINING FINISHED")

if __name__ == "__main__":
    Train()
