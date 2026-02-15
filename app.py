import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Predictions with Six Different ML Models")

uploaded_file = st.file_uploader(
    "Upload Test CSV (features only, no target column)",
    type="csv"
)

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if uploaded_file:
    X_test = pd.read_csv(uploaded_file)
    y_true = pd.read_csv("data/final_data/test.csv")["target"]

    with open(f"model/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    # AUC Calculation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = "Not available"

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write("AUC:", auc)
    st.write("Precision:", precision_score(y_true, y_pred))
    st.write("Recall:", recall_score(y_true, y_pred))
    st.write("F1 Score:", f1_score(y_true, y_pred))
    st.write("MCC:", matthews_corrcoef(y_true, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
