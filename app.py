import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction")

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV including target column)",
    type="csv"
)

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain 'target' column.")
    else:
        X_test = df.drop("target", axis=1)
        y_true = df["target"]

        with open(f"model/{model_name}.pkl", "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = None

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
        col2.metric("AUC", round(auc, 4) if auc else "N/A")
        col3.metric("Precision", round(precision_score(y_true, y_pred), 4))
        col4.metric("Recall", round(recall_score(y_true, y_pred), 4))
        col5.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
        col6.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 4))

        st.subheader("🧮 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.success("Model evaluation completed successfully!")
