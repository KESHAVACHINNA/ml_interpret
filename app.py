import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

import shap
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(layout="wide", page_title="ML Interpretability App")

st.title("💡 ML Interpretability App")
st.markdown("A Streamlit web app to visualize and understand how machine learning models make predictions using SHAP and ELI5.")

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map(lambda x: iris.target_names[x])
    X = df.drop(['target', 'target_names'], axis=1)
    y = df['target']
    return X, y, iris.feature_names, iris.target_names, df

X, y, feature_names, target_names, df_full = load_data()

# Sidebar for model selection and data display
st.sidebar.header("🧪 Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Classifier",
    ("Random Forest", "XGBoost", "LightGBM")
)

# Train the model
@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = train_model(model_name, X_train, y_train)

st.header("📊 Model Performance & Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Report")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

with col2:
    st.subheader("Sample Data")
    st.dataframe(df_full.head())
    st.write(f"Dataset Shape: {df_full.shape}")

st.markdown("---")

st.header("🔍 Global Feature Importance (SHAP Beeswarm Plot)")
st.write("The SHAP beeswarm plot shows how the top N features impact the model's output. Each point is a single prediction, and its horizontal location shows the impact of that feature on the prediction.")

# SHAP Global plot
try:
    if model_name == "Random Forest":
        explainer = shap.TreeExplainer(model)
    elif model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
    elif model_name == "LightGBM":
        explainer = shap.TreeExplainer(model)
    else:
        # For other models, use KernelExplainer (slower)
        explainer = shap.KernelExplainer(model.predict_proba, X_train)

    shap_values = explainer.shap_values(X_test)

    # If shap_values is a list (for multi-output models like multi-class classification),
    # take the average absolute SHAP values across classes for the beeswarm plot.
    if isinstance(shap_values, list):
        # For multi-class, shap_values is a list of arrays, one for each class
        # We need to combine them for a global view. A common way is to take the mean absolute SHAP value across classes.
        # Or, for a specific class, you can plot shap_values[class_index]
        # For simplicity in beeswarm, let's just use the first class's SHAP values or sum them up.
        # A better approach for multi-class beeswarm is to use `shap.summary_plot(shap_values, X_test)`
        # which handles the multi-class case internally.
        st.subheader(f"SHAP Beeswarm Plot (for all classes combined)")
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
        st.pyplot(fig_beeswarm)
    else:
        st.subheader(f"SHAP Beeswarm Plot")
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
        st.pyplot(fig_beeswarm)

except Exception as e:
    st.error(f"Error generating SHAP Beeswarm Plot: {e}")
    st.info("SHAP plots can sometimes be resource-intensive. Please try refreshing or selecting a different model.")


st.markdown("---")

st.header("🔬 Local Prediction Explanation (SHAP Force Plot)")
st.write("The SHAP force plot visualizes how features contribute to a single prediction, pushing the prediction higher (red) or lower (blue) than the base value.")

# SHAP Local plot
sample_index = st.slider("Select a sample from the test set to explain:", 0, len(X_test) - 1, 0)

try:
    # Get SHAP values for the selected instance
    if isinstance(shap_values, list):
        # For multi-class, you need to specify the class for the force plot
        # Let's assume we are explaining the prediction for the predicted class
        predicted_class_index = model.predict(X_test.iloc[[sample_index]])[0]
        shap_values_instance = shap_values[predicted_class_index][sample_index]
        expected_value_instance = explainer.expected_value[predicted_class_index]
    else:
        shap_values_instance = shap_values[sample_index]
        expected_value_instance = explainer.expected_value

    st.write(f"Explaining prediction for sample index: **{sample_index}**")
    st.write(f"Actual class: **{target_names[y_test.iloc[sample_index]]}**")
    st.write(f"Predicted class: **{target_names[model.predict(X_test.iloc[[sample_index]])[0]]}**")

    # Render force plot
    # Using st.components.v1.html to embed the JS visualization
    shap.initjs()
    force_plot_html = shap.force_plot(
        expected_value_instance,
        shap_values_instance,
        X_test.iloc[sample_index],
        feature_names=feature_names,
        matplotlib=False # Set to False to get HTML for Streamlit
    )
    st.components.v1.html(force_plot_html.html(), height=300)

except Exception as e:
    st.error(f"Error generating SHAP Force Plot: {e}")
    st.info("Please ensure a model is trained and a valid sample index is selected.")

st.markdown("---")

st.header("🧠 Feature Weights (ELI5 Permutation Importance)")
st.write("ELI5 Permutation Importance measures how much the model's prediction error increases when a feature's values are randomly shuffled. This indicates the feature's importance.")

try:
    perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
    # Convert ELI5 HTML to Streamlit compatible display
    eli5_html = eli5.show_weights(perm, feature_names=feature_names, target_names=target_names)._repr_html_()
    st.components.v1.html(eli5_html, height=500, scrolling=True)

except Exception as e:
    st.error(f"Error generating ELI5 Permutation Importance: {e}")
    st.info("Permutation Importance requires a fitted model and test data.")

st.markdown("---")
st.info("This app demonstrates basic ML interpretability. For production use, consider more robust error handling and performance optimizations.")
