import streamlit as st
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return clf, X_train, X_test, y_train, y_test, pred

def show_perf_metrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.subheader("Classification Report")
    st.sidebar.dataframe(pd.DataFrame(report).transpose().round(2))
    
    labels = sorted(list(set(y_test)))
    conf_matrix = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    st.sidebar.pyplot(plt)

def show_global_interpretation_shap(X_train, clf):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    st.subheader("SHAP Summary Plot")
    for i in range(len(shap_values)):
        st.markdown(f"**Class {i}**")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[i], X_train, plot_type="bar", show=False)
        st.pyplot(fig)

def main():
    st.title("🔍 ML Interpretability App")
    st.markdown("Explains model predictions using SHAP and performance metrics.")

    X, y, class_names = load_data()
    clf, X_train, X_test, y_train, y_test, pred = train_model(X, y)

    show_perf_metrics(y_test, pred)
    show_global_interpretation_shap(X_train, clf)

if __name__ == "__main__":
    main()
