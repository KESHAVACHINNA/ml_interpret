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

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    st.write(classification_report(y_test, pred, output_dict=False))

    # Global SHAP Explanation
    st.subheader("📈 Global Feature Importance (SHAP)")
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test)
    fig = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # Local SHAP Explanation
    st.subheader("🔬 Local Explanation (SHAP force plot)")
    idx = st.slider("Choose test sample index", 0, len(X_test) - 1, 0)
    st.write("Prediction:", clf.predict([X_test.iloc[idx]])[0], "Actual:", y_test.iloc[idx])
    st_shap(shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx]))

    # ELI5 Explanation
    st.subheader("🔍 Global Feature Weights (ELI5)")
    try:
        import eli5
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(clf, random_state=42).fit(X_test, y_test)
        st.components.v1.html(eli5.show_weights(perm, feature_names=X_test.columns.tolist()).data, height=400)
    except:
        st.warning("ELI5 not supported for this model.")

if __name__ == '__main__':
    main()
