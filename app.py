import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="ML Interpretability App")

st.title("💡 ML Interpretability App")
st.write("A Streamlit web app to visualize and understand how machine learning models make predictions using SHAP and ELI5.")

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map(lambda x: iris.target_names[x])
    return df, iris.feature_names, iris.target_names

df, feature_names, target_names = load_data()

# --- Sidebar for Model Selection and Training ---
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ("Random Forest", "XGBoost", "LightGBM")
)

# Train the model
X = df[feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = None
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_choice == "XGBoost":
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
elif model_choice == "LightGBM":
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

if st.sidebar.button("Train Model"):
    with st.spinner(f"Training {model_choice} model..."):
        model.fit(X_train, y_train)
    st.sidebar.success(f"{model_choice} Model Trained Successfully!")
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
else:
    if 'model' not in st.session_state:
        st.info("Train a model from the sidebar to see results.")

# --- Main Content Area ---

if 'model' in st.session_state:
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    st.subheader("📊 Model Performance")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.json(report)

    st.subheader("📋 Sample Data")
    st.dataframe(X_test.head())

    st.subheader("🔍 Global Feature Importance (SHAP Beeswarm Plot)")
    st.write("The SHAP beeswarm plot shows the overall impact of each feature on the model's predictions.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP beeswarm plot for multiclass classification
    # For multiclass, shap_values is a list of arrays, one for each class.
    # We can visualize the mean absolute SHAP value across all classes for each feature,
    # or choose a specific class. For simplicity, we'll plot for a chosen class.
    # A more advanced app might allow selection of class for beeswarm.
    # Here, we'll take the sum of absolute SHAP values across classes to represent overall importance.
    
    # Calculate mean absolute SHAP value across all classes for each feature
    if isinstance(shap_values, list): # For multiclass models
        mean_abs_shap = np.mean(np.abs(np.array(shap_values)), axis=0)
        shap.summary_plot(mean_abs_shap, X_test, feature_names=feature_names, plot_type="beeswarm", show=False)
    else: # For binary or single output models (though Iris is multiclass)
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="beeswarm", show=False)
    
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf() # Clear the plot to prevent overlapping

    st.subheader("🔬 Local Prediction Explanation (SHAP Force Plot)")
    st.write("The SHAP force plot visualizes how features contribute to a single prediction.")
    
    instance_index = st.slider("Select an instance from the test set", 0, len(X_test) - 1, 0)
    selected_instance = X_test.iloc[[instance_index]]

    # For multiclass, explainer.shap_values returns a list of arrays (one per class).
    # We need to select the shap_values for the predicted class.
    predicted_class_index = model.predict(selected_instance)[0]

    # Initialize JS for SHAP
    shap.initjs()
    
    if isinstance(shap_values, list): # Multiclass
        # Ensure we have the explainer for the chosen class to get base_values correctly
        explainer_for_class = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        # Base value for the specific predicted class
        base_value_for_class = explainer_for_class.expected_value[predicted_class_index]
        
        # SHAP values for the specific predicted class and instance
        shap_values_instance_for_class = explainer_for_class.shap_values(selected_instance)[predicted_class_index]

        # Use st.components.v1.html to render the force plot (requires JS)
        force_plot_html = shap.force_plot(
            base_value_for_class, 
            shap_values_instance_for_class, 
            selected_instance,
            show=False,
            matplotlib=False # Set to False to get HTML for Streamlit
        )
    else: # Binary or single output (not applicable to Iris, but for completeness)
        force_plot_html = shap.force_plot(
            explainer.expected_value,
            explainer.shap_values(selected_instance),
            selected_instance,
            show=False,
            matplotlib=False
        )

    # Convert the HTML object to a string and embed it
    st.components.v1.html(force_plot_html.html(), height=300)

    st.subheader("🧠 Feature Weights (ELI5 Permutation Importance)")
    st.write("ELI5 permutation importance shows how much the model's performance decreases when a feature is randomly shuffled (permuted).")
    
    # For multiclass, ELI5 needs a specific class to compute importance for or it averages.
    # Let's compute for all classes and display a table.
    perm_importance = PermutationImportance(model, random_state=42)
    perm_importance.fit(X_test, y_test)

    eli5_weights = eli5.format_as_html(eli5.show_weights(perm_importance, feature_names=feature_names, target_names=target_names))
    st.components.v1.html(eli5_weights, height=500, scrolling=True)

else:
    st.info("Train a model from the sidebar to visualize interpretability results.")

st.markdown("---")
st.markdown("💡 Features:")
st.markdown("- 🧪 Select and train classifiers (Random Forest, XGBoost, LightGBM) on the Iris dataset.")
st.markdown("- 📊 View classification reports and sample data.")
st.markdown("- 🔍 Visualize global feature importance using **SHAP beeswarm plots**.")
st.markdown("- 🔬 Understand local predictions using **SHAP force plots**.")
st.markdown("- 🧠 Analyze feature weights with **ELI5 permutation importance**.")
