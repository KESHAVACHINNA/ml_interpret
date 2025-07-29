import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components

# Removed deprecated config option
# st.set_option('deprecation.showPyplotGlobalUse', False)

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
            return data.data, data.target

    def get_classifier(name):
        if name == "Random Forest":
            return RandomForestClassifier()
        elif name == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif name == "LightGBM":
            return LGBMClassifier()

    X, y = get_dataset(dataset_name)
    clf = get_classifier(classifier_name)

    st.write("## Sample of the Data")
    st.dataframe(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    st.write("### Model Accuracy")
    st.text(classification_report(y_test, pred))

    # SHAP Global Explanation
    st.subheader("📈 Global Feature Importance (SHAP)")
    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches="tight")
    plt.clf()

    # SHAP Local Explanation
    st.subheader("🔬 Local Explanation (SHAP force plot)")
    idx = st.slider("Choose test sample index", 0, len(X_test) - 1, 0)
    pred_class = clf.predict([X_test.iloc[idx]])[0]
    st.write("Prediction:", pred_class, "| Actual:", y_test.iloc[idx])
    st_shap(shap.force_plot(explainer.expected_value, shap_values[idx].values, X_test.iloc[idx]))

    # ELI5 Explanation
    st.subheader("🔍 Global Feature Weights (ELI5)")
    try:
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(clf, random_state=42).fit(X_test, y_test)
        html = eli5.show_weights(perm, feature_names=X_test.columns.tolist()).data
        components.html(html, height=400, scrolling=True)
    except Exception as e:
        st.warning("ELI5 not supported for this model. Error: " + str(e))

if __name__ == '__main__':
    main()
