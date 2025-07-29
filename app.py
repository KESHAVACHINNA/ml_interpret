import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Setup Streamlit
st.set_page_config(page_title="ML Interpretation App", layout="wide")
st.title("📊 BlackBox ML Classifier Explainer")

# Function to load and process uploaded data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Sidebar upload
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_data(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select target and features
    target = st.sidebar.selectbox("Select Target Column", df.columns)
    features = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        # Encode target if categorical
        if y.dtype == 'O':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        # Metrics
        st.subheader("Classification Report")
        report = classification_report(y_test, pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
        st.pyplot(plt)

        # SHAP global
        st.subheader("SHAP Global Feature Importance")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

        # SHAP local
        st.subheader("SHAP Local Explanation (Single Prediction)")
        index = st.slider("Choose Index for Explanation", 0, len(X_test)-1, 0)

        fig2 = shap.plots.waterfall(shap.Explanation(values=shap_values[1][index],
                                                     base_values=explainer.expected_value[1],
                                                     data=X_test.iloc[index]), show=False)
        st.pyplot(fig2)
else:
    st.info("Upload a CSV file to start.")
