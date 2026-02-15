import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Load test data
df = pd.read_csv("../data/test.csv")
X_test = df.drop("target", axis=1)
y_test = df["target"]

model_files = [
    "logistic.pkl",
    "decision_tree.pkl",
    "knn.pkl",
    "naive_bayes.pkl",
    "random_forest.pkl",
    "xgboost.pkl"
]

results = []

for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": file.replace(".pkl", ""),
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    })

print(pd.DataFrame(results))
