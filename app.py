import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    target_labels = data.target_names
    return X, y, target_labels

# Model Training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return clf, X_test, y_test, pred

# Performance Metrics
def show_perf_metrics(y_test, pred):
    conf_matrix = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.json(report)

# Global Interpretation

def show_global_interpretation(clf, X_test):
    explainer = shap.Explainer(clf, X_test)
    shap_values = explainer(X_test)
    st.subheader("SHAP Global Explanation (Beeswarm Plot)")
    try:
        fig = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating SHAP beeswarm plot: {e}")

# Local Interpretation

def show_local_interpretation_shap(clf, X_test, pred, target_labels, idx):
    explainer = shap.Explainer(clf, X_test)
    shap_values = explainer(X_test)
    try:
        st.subheader(f"SHAP Force Plot for sample {idx}")
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
        shap_html = shap.plots.force(
            expected_value,
            shap_values[idx].values,
            X_test.iloc[idx],
            matplotlib=False,
            show=False
        )
        st_shap(shap_html)
    except IndexError:
        st.warning("Selected index is out of range for SHAP values.")

# Display SHAP HTML
from streamlit.components.v1 import html

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height or 500)

def show_local_interpretation(X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework):
    if len(X_test) > 0:
        slider_idx = st.slider("Pick test sample index", 0, len(X_test) - 1)
        show_local_interpretation_shap(clf, X_test, pred, target_labels, slider_idx)
    else:
        st.warning("Test set is empty. Cannot show local interpretation.")

# Main App

def main():
    st.title("Blackbox ML Classifiers Visually Explained")

    X, y, target_labels = load_data()
    clf, X_test, y_test, pred = train_model(X, y)

    st.header("Model Performance")
    show_perf_metrics(y_test, pred)

    st.header("Model Interpretation")
    st.subheader("Global Interpretation")
    show_global_interpretation(clf, X_test)

    st.subheader("Local Interpretation")
    show_local_interpretation(
        X_test, y_test, clf, pred, target_labels, X.columns.tolist(), "RandomForest", "TreeBased"
    )

if __name__ == "__main__":
    main()
