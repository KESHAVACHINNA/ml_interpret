import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import base64
import io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Interpretability App", layout="wide")
st.title("🧠 Advanced ML Model Interpretability Tool")

# Sidebar Inputs
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Function to preprocess the data
def preprocess_data(df):
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

# Function to get classifier
@st.cache_resource
def get_classifier(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100)
    elif name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif name == "LightGBM":
        return LGBMClassifier()
    elif name == "GradientBoosting":
        return GradientBoostingClassifier()
    elif name == "LogisticRegression":
        return LogisticRegression(max_iter=1000)

# Main logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    
    target = st.sidebar.selectbox("Select Target Column", df.columns)
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_name = st.sidebar.selectbox("Choose a Model", ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "LogisticRegression"])
    clf = get_classifier(model_name)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, pred))

    st.subheader("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("📈 ROC Curve")
    if len(np.unique(y)) == 2:
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📌 SHAP Global Explanation")
    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("🔬 SHAP Dependence Plot")
    selected_feat = st.selectbox("Feature for SHAP Dependence", df.columns[:-1])
    fig = plt.figure()
    shap.dependence_plot(selected_feat, shap_values.values, X_test, show=False)
    st.pyplot(fig)

    st.subheader("📌 SHAP Force Plot (Single Prediction)")
    idx = st.slider("Choose Sample Index", 0, X_test.shape[0] - 1, 0)
    shap_html = shap.plots.force(explainer.expected_value[0], shap_values.values[idx], X_test[idx], matplotlib=False)
    st.components.v1.html(shap.save_html(shap_html), height=300)

    st.subheader("📋 Feature Importances")
    importance = pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
    importance = importance.sort_values(by="importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x="importance", y="feature", data=importance, ax=ax)
    st.pyplot(fig)

    st.subheader("📎 ELI5 Explanation")
    with st.expander("View ELI5 Explanation"):
        html_obj = eli5.show_weights(clf, feature_names=features)
        raw_html = html_obj.data
        st.components.v1.html(raw_html, scrolling=True, height=400)

    st.success("✅ Interpretation complete! You can explore more by changing models or features.")

else:
    st.info("👈 Upload a CSV file to get started!")
