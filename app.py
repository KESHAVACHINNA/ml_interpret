import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import shap
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import io # Required for capturing matplotlib figures

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="ML Interpretability App")

st.title("💡 ML Interpretability App")
st.markdown("A Streamlit web app to visualize and understand how machine learning models make predictions using SHAP and ELI5.")
st.markdown("---")

# --- Data Loading ---
@st.cache_data
def load_iris_data():
    """Loads the Iris dataset and returns features, target, and full DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map(lambda x: iris.target_names[x])
    X = df.drop(['target', 'target_names'], axis=1)
    y = df['target']
    return X, y, iris.feature_names, iris.target_names, df

X, y, feature_names, target_names, df_full = load_iris_data()

# --- Sidebar for Model Configuration ---
st.sidebar.header("🧪 Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Classifier",
    ("Random Forest", "XGBoost", "LightGBM")
)

# --- Model Training ---
@st.cache_resource # Use st.cache_resource for models to avoid retraining on every rerun
def train_selected_model(model_name, X_train_data, y_train_data):
    """Trains the selected machine learning model."""
    st.sidebar.write(f"Training {model_name}...")
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif model_name == "XGBoost":
        # Suppress the UserWarning about use_label_encoder
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    
    model.fit(X_train_data, y_train_data)
    st.sidebar.success(f"{model_name} trained successfully!")
    return model

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model based on user selection
model = train_selected_model(model_name, X_train, y_train)

# --- Model Performance & Data Overview Section ---
st.header("📊 Model Performance & Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Report")
    y_pred = model.predict(X_test)
    
    # Calculate and display common metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision (weighted):** {precision:.4f}")
    st.write(f"**Recall (weighted):** {recall:.4f}")
    st.write(f"**F1-Score (weighted):** {f1:.4f}")

    st.markdown("---")
    st.write("Full Classification Report:")
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

with col2:
    st.subheader("Sample Data")
    st.dataframe(df_full.head(10)) # Display a few more rows
    st.write(f"**Dataset Shape:** {df_full.shape}")
    st.write(f"**Features:** {', '.join(feature_names)}")
    st.write(f"**Target Classes:** {', '.join(target_names)}")

st.markdown("---")

# --- Global Feature Importance (SHAP Beeswarm Plot) ---
st.header("🔍 Global Feature Importance (SHAP Beeswarm Plot)")
st.write("The SHAP beeswarm plot shows how the top N features impact the model's output. Each point is a single prediction, and its horizontal location shows the impact of that feature on the prediction.")

try:
    # Initialize SHAP explainer based on model type
    if model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback for other models (less efficient)
        explainer = shap.KernelExplainer(model.predict_proba, X_train)

    # Calculate SHAP values for the test set
    # For multi-class classification, shap_values will be a list of arrays, one for each class.
    shap_values = explainer.shap_values(X_test)

    # Plotting the beeswarm plot
    # shap.summary_plot handles multi-class SHAP values internally for beeswarm plot
    st.subheader(f"SHAP Beeswarm Plot for {model_name}")
    
    # Use plt.figure() to get a new figure and prevent plots from overlapping
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(12, 7)) # Adjust figure size for better visibility
    
    # Make sure to pass feature_names to summary_plot for better labels
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="beeswarm", show=False)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    st.pyplot(fig_beeswarm)
    plt.close(fig_beeswarm) # Close the figure to free memory

except Exception as e:
    st.error(f"Error generating SHAP Beeswarm Plot: {e}")
    st.info("SHAP plots can be computationally intensive. Ensure you have enough resources.")
    st.info("If using a custom model, ensure it's compatible with TreeExplainer or KernelExplainer.")

st.markdown("---")

# --- Local Prediction Explanation (SHAP Force Plot) ---
st.header("🔬 Local Prediction Explanation (SHAP Force Plot)")
st.write("The SHAP force plot visualizes how features contribute to a single prediction, pushing the prediction higher (red) or lower (blue) than the base value.")

# Allow user to select a sample index from the test set
sample_index = st.slider("Select a sample from the test set to explain:", 0, len(X_test) - 1, 0)

try:
    # Get the specific instance from the test set
    selected_instance = X_test.iloc[sample_index]

    # Determine predicted class for the selected instance
    predicted_class_for_instance = model.predict(selected_instance.to_frame().T)[0]

    # Get SHAP values and expected value for the selected instance and its predicted class
    if isinstance(shap_values, list):
        # For multi-class, select SHAP values and expected value corresponding to the predicted class
        shap_values_instance = shap_values[predicted_class_for_instance][sample_index]
        expected_value_instance = explainer.expected_value[predicted_class_for_instance]
    else:
        # For binary or regression, shap_values is a single array
        shap_values_instance = shap_values[sample_index]
        expected_value_instance = explainer.expected_value

    st.write(f"**Explaining prediction for sample index:** `{sample_index}`")
    st.write(f"**Actual class:** `{target_names[y_test.iloc[sample_index]]}`")
    st.write(f"**Predicted class:** `{target_names[predicted_class_for_instance]}`")

    # Render the force plot using Streamlit components HTML
    shap.initjs() # Initialize SHAP's JavaScript for rendering
    force_plot_html = shap.force_plot(
        expected_value_instance,
        shap_values_instance,
        selected_instance, # Pass the actual data instance
        feature_names=feature_names,
        matplotlib=False # Crucial for getting HTML output
    )
    # Use io.StringIO to capture the HTML output from force_plot
    shap_html = io.StringIO()
    shap.save_html(shap_html, force_plot_html)
    st.components.v1.html(shap_html.getvalue(), height=350, scrolling=True)

except Exception as e:
    st.error(f"Error generating SHAP Force Plot: {e}")
    st.info("Ensure a model is trained and a valid sample index is selected. This plot can also be sensitive to SHAP version and model type.")

st.markdown("---")

# --- Feature Weights (ELI5 Permutation Importance) ---
st.header("🧠 Feature Weights (ELI5 Permutation Importance)")
st.write("ELI5 Permutation Importance measures how much the model's prediction error increases when a feature's values are randomly shuffled. This indicates the feature's importance.")

try:
    # Initialize PermutationImportance and fit it to the model and test data
    perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
    
    # Generate ELI5 weights HTML
    # Ensure target_names are passed for multi-class classification
    eli5_html = eli5.show_weights(perm, feature_names=feature_names, target_names=target_names)._repr_html_()
    
    # Embed the ELI5 HTML into Streamlit
    st.components.v1.html(eli5_html, height=500, scrolling=True)

except Exception as e:
    st.error(f"Error generating ELI5 Permutation Importance: {e}")
    st.info("Permutation Importance requires a fitted model and test data. Check if your model is properly trained.")

st.markdown("---")
st.info("This app demonstrates basic ML interpretability. For production use, consider more robust error handling, performance optimizations, and handling larger datasets.")
