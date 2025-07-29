import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Advanced ML Interpretability", layout="wide")
st.title("🔍 Advanced Machine Learning Interpretability Dashboard")

# Upload Dataset
st.sidebar.header("📁 Upload CSV Dataset")
file = st.sidebar.file_uploader("Choose your dataset", type=["csv"])

# Preprocess data
def preprocess(df):
    df = df.dropna()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Get Classifier
def get_model(model_name):
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=100)
    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_name == "LightGBM":
        return LGBMClassifier()
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier()
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000)

# Main App
if file is not None:
    df = pd.read_csv(file)
    df = preprocess(df)

    st.subheader("🔬 Dataset Preview")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    features = [f for f in df.columns if f != target]
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model_choice = st.sidebar.selectbox("⚙️ Choose Classifier", ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "LogisticRegression"])
    model = get_model(model_choice)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧱 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Cross Validation Scores")
    scores = cross_val_score(model, X_scaled, y, cv=5)
    st.write(f"Average Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    if len(np.unique(y)) == 2:
        st.subheader("📈 ROC Curve")
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📌 SHAP Summary Plot")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("📌 SHAP Dependence Plot")
    feature_name = st.selectbox("Select Feature", features)
    fig, ax = plt.subplots()
    shap.dependence_plot(feature_name, shap_values.values, X_test, show=False)
    st.pyplot(fig)

    st.subheader("📌 SHAP Force Plot")
    idx = st.slider("Sample Index", 0, len(X_test) - 1, 0)
    force_plot_html = shap.plots.force(explainer.expected_value[0], shap_values.values[idx], X_test[idx], matplotlib=False)
    st.components.v1.html(shap.save_html(force_plot_html), height=300)

    if hasattr(model, "feature_importances_"):
        st.subheader("📌 Feature Importances")
        importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        importances = importances.sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importances, ax=ax)
        st.pyplot(fig)

    st.subheader("📎 ELI5 Model Explanation")
    with st.expander("Show Weights"):
        html_expl = eli5.show_weights(model, feature_names=features)
        st.components.v1.html(html_expl.data, height=400)

    st.success("✅ Model interpretation complete!")
else:
    st.warning("📤 Upload a dataset to begin.")
