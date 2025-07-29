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
    """Loads the Iris dataset and returns it as a DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map(lambda x: iris.target_names[x])
    return df, iris.feature_names, iris.target_names, iris.target_names.tolist()

df, feature_names, target_names_array, target_names_list = load_data()

# --- Sidebar for Model Selection and Training ---
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ("Random Forest", "XGBoost", "LightGBM")
)

# Sliders for hyperparameters
n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
learning_rate = st.sidebar.slider("Learning Rate (XGBoost/LightGBM only)", 0.01, 0.3, 0.1, 0.01)

# Train the model
X = df[feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = None
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_choice == "XGBoost":
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
elif model_choice == "LightGBM":
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

# Button to train the model
if st.sidebar.button("Train Model"):
    with st.spinner(f"Training {model_choice} model..."):
        try:
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['model_trained'] = True # Flag to indicate model is trained
            st.sidebar.success(f"{model_choice} Model Trained Successfully!")
        except Exception as e:
            st.sidebar.error(f"Error training model: {e}")
            st.session_state['model_trained'] = False
else:
    # Initialize the flag if not present
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if not st.session_state['model_trained']:
        st.info("Train a model from the sidebar to see results.")

# --- Main Content Area ---

if st.session_state.get('model_trained', False):
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    st.subheader("📊 Model Performance")
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names_list, output_dict=True)
        st.json(report)
    except Exception as e:
        st.error(f"Error generating classification report: {e}")

    st.subheader("📋 Sample Data")
    st.dataframe(X_test.head())

    # --- SHAP Global Feature Importance ---
    st.subheader("🔍 Global Feature Importance (SHAP Beeswarm Plot)")
    st.write("The SHAP beeswarm plot shows the overall impact of each feature on the model's predictions.")
    try:
        explainer = shap.TreeExplainer(model)
        # For multiclass, shap_values will be a list of arrays.
        # We need to compute SHAP values for the entire X_test to get a good beeswarm plot.
        shap_values = explainer.shap_values(X_test)

        # Plotting the beeswarm for multiclass:
        # shap.summary_plot can handle a list of shap_values for multiclass.
        # It will either plot a combined view or allow selection of a class.
        # By default, for multiclass, it often shows the overall importance.
        # Using the base `shap.summary_plot` directly should work well.
        
        plt.figure(figsize=(10, 6))
        # Ensure correct passing of feature names for multiclass
        if isinstance(shap_values, list): # Multiclass output
            # For multiclass, plotting the sum of absolute SHAP values across classes is a common way
            # to get a single 'global importance' view.
            # However, shap.summary_plot handles a list of arrays for different classes effectively.
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="beeswarm", show=False)
        else: # Binary or regression
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="beeswarm", show=False)
        
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf() # Clear the plot to prevent overlapping issues in Streamlit
    except Exception as e:
        st.error(f"Error generating SHAP Beeswarm Plot: {e}")
        st.warning("SHAP plots might require specific versions of libraries or might not work perfectly with all model types.")

    # --- SHAP Local Prediction Explanation ---
    st.subheader("🔬 Local Prediction Explanation (SHAP Force Plot)")
    st.write("The SHAP force plot visualizes how features contribute to a single prediction.")
    
    if not X_test.empty:
        instance_index = st.slider("Select an instance from the test set", 0, len(X_test) - 1, 0)
        selected_instance = X_test.iloc[[instance_index]]
        
        try:
            # Initialize JS for SHAP
            shap.initjs()

            # The TreeExplainer works well for tree-based models.
            # For other models, you might need shap.KernelExplainer or shap.DeepExplainer.
            explainer_local = shap.TreeExplainer(model)
            
            # Get SHAP values for the selected instance
            shap_values_instance = explainer_local.shap_values(selected_instance)

            # Get the base value (expected value)
            expected_value = explainer_local.expected_value
            
            # Handle multiclass for force plot
            if isinstance(shap_values_instance, list): # Multiclass
                # For force plot, we typically show one class.
                # Let's show the force plot for the predicted class.
                predicted_class_index = model.predict(selected_instance)[0]
                
                # Ensure expected_value is also for the specific class
                if isinstance(expected_value, np.ndarray):
                    expected_value_for_plot = expected_value[predicted_class_index]
                else:
                    expected_value_for_plot = expected_value # If it's a single value already

                force_plot_html = shap.force_plot(
                    expected_value_for_plot,
                    shap_values_instance[predicted_class_index], # SHAP values for the predicted class
                    selected_instance,
                    show=False,
                    matplotlib=False # Crucial for getting HTML for Streamlit
                )
            else: # Binary or regression
                force_plot_html = shap.force_plot(
                    expected_value,
                    shap_values_instance,
                    selected_instance,
                    show=False,
                    matplotlib=False
                )
            
            # Render the HTML component
            st.components.v1.html(force_plot_html.html(), height=350)
            
        except Exception as e:
            st.error(f"Error generating SHAP Force Plot: {e}")
            st.warning("SHAP Force Plots can be sensitive to data types or explainer types. Ensure your SHAP library is updated.")
    else:
        st.warning("No test data available to generate a force plot.")


    # --- ELI5 Permutation Importance ---
    st.subheader("🧠 Feature Weights (ELI5 Permutation Importance)")
    st.write("ELI5 permutation importance shows how much the model's performance decreases when a feature is randomly shuffled (permuted).")
    
    try:
        # Initialize PermutationImportance
        perm_importance = PermutationImportance(model, random_state=42)
        
        # Fit on the test data
        perm_importance.fit(X_test, y_test)

        # Generate the ELI5 HTML output
        # For multiclass, eli5.show_weights might show weights for each class
        # or an average. Passing target_names helps.
        eli5_weights_html = eli5.format_as_html(
            eli5.show_weights(
                perm_importance,
                feature_names=feature_names,
                target_names=target_names_list # Use the list of target names
            )
        )
        
        # Display the HTML in Streamlit
        st.components.v1.html(eli5_weights_html, height=500, scrolling=True)

    except Exception as e:
        st.error(f"Error generating ELI5 Permutation Importance: {e}")
        st.warning("ELI5 might have issues with specific model types or if it can't determine feature importance.")

else:
    st.info("Train a model from the sidebar to visualize interpretability results.")

st.markdown("---")
st.markdown("💡 Features:")
st.markdown("- 🧪 Select and train classifiers (Random Forest, XGBoost, LightGBM) on the Iris dataset.")
st.markdown("- 📊 View classification reports and sample data.")
st.markdown("- 🔍 Visualize global feature importance using **SHAP beeswarm plots**.")
st.markdown("- 🔬 Understand local predictions using **SHAP force plots**.")
st.markdown("- 🧠 Analyze feature weights with **ELI5 permutation importance**.")
