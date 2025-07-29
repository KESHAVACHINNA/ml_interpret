import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data():
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Dataset:")
        st.write(data.head())
        return data
    else:
        st.info("Awaiting for CSV file to be uploaded.")
        return None


def build_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def show_perf_metrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.subheader("Classification Report")
    st.sidebar.dataframe(pd.DataFrame(report).round(2).transpose())

    conf_matrix = confusion_matrix(y_test, pred, labels=list(set(y_test)))
    sns.set(font_scale=1.2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        square=True,
        annot=True,
        annot_kws={"size": 12},
        cmap="YlGnBu",
        cbar=False,
        fmt='d'
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.sidebar.subheader("Confusion Matrix")
    st.sidebar.pyplot()


def show_global_interpretation_shap(X_train, clf):
    st.subheader("Global Interpretation using SHAP")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):  # multiclass
        for i, class_shap_values in enumerate(shap_values):
            st.write(f"SHAP Summary Plot for Class {i}")
            shap.summary_plot(class_shap_values, X_train, plot_type="bar", show=False)
            st.pyplot()
    else:  # binary or regression
        st.write("SHAP Summary Plot")
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot()


def main():
    st.title("BlackBox ML Classifiers Visually Explained")
    st.markdown("A Streamlit app to interpret machine learning classifiers using SHAP.")

    data = load_data()
    if data is not None:
        if st.sidebar.checkbox("Run Model and Explain", value=True):
            target_column = st.sidebar.selectbox("Select Target Column", data.columns)

            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = build_model(X_train, y_train)
            pred = clf.predict(X_test)

            show_perf_metrics(y_test, pred)
            show_global_interpretation_shap(X_train, clf)


if __name__ == '__main__':
    main()
