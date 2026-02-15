import os
import pandas as pd
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load training data
print("Loading training data.")
df = pd.read_csv("../data/train.csv")

X = df.drop("target", axis=1)
y = df["target"]

print(f"Data loaded and Shape of data is: {df.shape}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {
    "logistic": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "decision_tree": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "knn": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "naive_bayes": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("model", GaussianNB())
    ]),

    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),

    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss"
        ))
    ])
}

print("\n Starting model training\n")
for name, model in models.items():
    start_time = datetime.now()

    print(f"Training {name.replace('_', ' ').title()} model")
    model.fit(X, y)

    model_path = os.path.join(BASE_DIR, f"{name}.pkl")
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"{name.replace('_', ' ').title()} training completed")
    print(f"Model saved as: {model_path}")
    print(f"Training time: {duration:.4f} seconds\n")

print("All models are trained successfully")
