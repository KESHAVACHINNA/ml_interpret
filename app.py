import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components

def st_shap(plot, height=300):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    st.title("🔍 ML Interpretability App")

    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris",))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("Random Forest", "XGBoost", "LightGBM"))

    def get_dataset(name):
        if name == "Iris":
            data = load_iris(as_frame=True)
            X, y = data.data, data.target
        return X, y

    def get_classifier(name):
        if name == "Random Forest":
            clf = RandomForestClassifier()
        elif name == "XGBoost":
            clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif name == "LightGBM":
            clf = LGBMClassifier()
        return clf

    X, y = get_dataset(dataset_name)
    clf = get_classifier(classifier_name)

    st.write("## Sample of the Data")
    st.dataframe(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    st.write("### Model Accuracy")
    st.text(classification_report(y_test, pred))

    # SHAP Global Explanation
    st.subheader("📈 SHAP Global Explanation (Beeswarm Plot)")
    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)

    if len(shap_values.shape) == 3:
        selected_class = st.selectbox("Select class for SHAP beeswarm plot", list(range(shap_values.shape[2])))
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values[:, :, selected_class], ax=ax, show=False)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, ax=ax, show=False)
        st.pyplot(fig)

    # SHAP Local Explanation
    st.subheader("🔬 SHAP Local Explanation (Force Plot)")
    idx = st.slider("Choose test sample index", 0, len(X_test) - 1, 0)
    st.write("Prediction:", clf.predict([X_test.iloc[idx]])[0], "Actual:", y_test.iloc[idx])
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[idx][:, selected_class], X_test.iloc[idx]))

    # ELI5 Explanation
    st.subheader("🔍 ELI5 Global Feature Weights")
    try:
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(clf, random_state=42).fit(X_test, y_test)
        html = eli5.show_weights(perm, feature_names=X_test.columns.tolist()).data
        st.components.v1.html(html, height=400)
    except Exception as e:
        st.warning("ELI5 not supported for this model.")

if __name__ == '__main__':
    main()
