import streamlit as st
import pandas as pd
import numpy as np
import shap
import eli5
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import streamlit.components.v1 as components
import io # Used for capturing HTML output from SHAP

warnings.filterwarnings("ignore") # Suppress warnings, e.g., from XGBoost's use_label_encoder

st.set_page_config(page_title="Advanced ML Interpretability", layout="wide")
st.title("🔍 Advanced Machine Learning Interpretability Dashboard")
st.markdown("Upload your CSV dataset to explore model performance and interpret predictions using SHAP and ELI5.")
st.markdown("---")

# --- Data Upload and Preprocessing ---
st.sidebar.header("📁 Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose your dataset", type=["csv"])

@st.cache_data
def preprocess_data(df_input):
    """
    Preprocesses the dataframe by dropping NA values and label encoding object columns.
    Returns the processed DataFrame.
    """
    df_processed = df_input.copy()
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    if len(df_processed) < initial_rows:
        st.sidebar.warning(f"Dropped {initial_rows - len(df_processed)} rows with missing values.")

    for col in df_processed.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        st.sidebar.info(f"Label encoded column: '{col}'")
    return df_processed

# --- Model Selection ---
@st.cache_resource
def get_model(model_name):
    """Returns an initialized classifier based on the selected name."""
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        # eval_metric='logloss' is for binary classification, 'mlogloss' for multi-class
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == "LightGBM":
        return LGBMClassifier(random_state=42)
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42)
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42)

# --- Main Application Logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)

    st.subheader("🔬 Dataset Preview")
    st.dataframe(df.head())

    # Target and Feature Selection
    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    
    # Ensure target column is not empty
    if target not in df.columns:
        st.error("Selected target column not found in the dataset.")
        st.stop()

    features = [f for f in df.columns if f != target]
    if not features:
        st.error("No feature columns available after selecting target. Please upload a dataset with at least two columns.")
        st.stop()

    X = df[features]
    y = df[target]

    # Check for binary classification for ROC curve
    is_binary_classification = len(np.unique(y)) == 2

    # Split data BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if is_binary_classification else None)

    # Scale numerical features AFTER splitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with original feature names for SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

    model_choice = st.sidebar.selectbox("⚙️ Choose Classifier", ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "LogisticRegression"])
    
    # Train model
    model = get_model(model_choice)
    model.fit(X_train_scaled_df, y_train) # Use scaled training data
    y_pred = model.predict(X_test_scaled_df) # Use scaled test data

    st.subheader("📋 Model Performance Metrics")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧱 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    plt.close(fig_cm) # Close figure

    st.subheader("📊 Cross Validation Scores")
    # Use scaled full data for cross-validation
    scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    st.write(f"Average Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # ROC Curve (only for binary classification)
    if is_binary_classification:
        st.subheader("📈 ROC Curve")
        try:
            y_proba = model.predict_proba(X_test_scaled_df)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
            ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)
            plt.close(fig_roc) # Close figure
        except Exception as e:
            st.warning(f"Could not generate ROC Curve: {e}. This might happen if `predict_proba` is not available or for multi-class problems.")
    else:
        st.info("ROC Curve is only applicable for binary classification problems.")

    st.markdown("---")
    st.header("📌 SHAP (SHapley Additive exPlanations)")

    @st.cache_resource # Cache explainer and shap_values as they can be expensive
    def get_shap_values(model, X_train_scaled_df, X_test_scaled_df):
        """Initializes SHAP explainer and computes SHAP values."""
        # For tree models, TreeExplainer is faster. For others, KernelExplainer or Explainer.
        if model_choice in ["RandomForest", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
        else:
            # Using a subset of X_train for KernelExplainer for performance
            # if X_train_scaled_df.shape[0] > 100:
            #     background_data = shap.sample(X_train_scaled_df, 100)
            # else:
            #     background_data = X_train_scaled_df
            # explainer = shap.KernelExplainer(model.predict_proba, background_data)
            # For simplicity, using shap.Explainer which handles model types
            explainer = shap.Explainer(model, X_train_scaled_df)

        # Calculate SHAP values
        # For multi-class, shap_values will be a shap.Explanation object with multiple outputs
        shap_values = explainer(X_test_scaled_df)
        return explainer, shap_values

    explainer, shap_values = get_shap_values(model, X_train_scaled_df, X_test_scaled_df)

    st.subheader("Global Feature Importance (SHAP Bar Plot)")
    st.write("This plot shows the average absolute SHAP value for each feature, indicating its overall importance.")
    try:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        # Use shap_values.values for the bar plot if multi-class, it averages automatically
        shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False, feature_names=features)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    except Exception as e:
        st.error(f"Error generating SHAP Bar Plot: {e}")
        st.info("Ensure the model is compatible with SHAP Explainer.")

    st.subheader("Global Feature Impact (SHAP Beeswarm Plot)")
    st.write("Each point represents a single prediction. Its horizontal position shows the impact of that feature on the prediction. Color indicates feature value (red=high, blue=low).")
    try:
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(12, 7))
        shap.summary_plot(shap_values, X_test_scaled_df, plot_type="beeswarm", show=False, feature_names=features)
        plt.tight_layout()
        st.pyplot(fig_beeswarm)
        plt.close(fig_beeswarm)
    except Exception as e:
        st.error(f"Error generating SHAP Beeswarm Plot: {e}")
        st.info("Beeswarm plots can be resource-intensive. Try with a smaller test set if issues persist.")

    st.subheader("Feature Dependence Plot")
    st.write("Shows the effect of a single feature on the prediction, colored by another feature to reveal interactions.")
    if len(features) > 0:
        feature_for_dependence = st.selectbox("Select Feature for Dependence Plot", features, key="dependence_feature")
        
        # Select an interacting feature (e.g., the most important one besides the selected feature)
        # Or let the user select it
        if len(features) > 1:
            other_features = [f for f in features if f != feature_for_dependence]
            # Simple heuristic: pick the first other feature as the interaction feature
            interaction_feature = st.selectbox("Select Interaction Feature (Optional)", ["None"] + other_features, key="interaction_feature")
        else:
            interaction_feature = "None"

        try:
            fig_dep, ax_dep = plt.subplots(figsize=(8, 6))
            if interaction_feature != "None":
                shap.dependence_plot(
                    feature_for_dependence,
                    shap_values.values, # Use shap_values.values for dependence plot
                    X_test_scaled_df,
                    interaction_index=interaction_feature,
                    show=False,
                    feature_names=features
                )
            else:
                shap.dependence_plot(
                    feature_for_dependence,
                    shap_values.values, # Use shap_values.values for dependence plot
                    X_test_scaled_df,
                    show=False,
                    feature_names=features
                )
            plt.tight_layout()
            st.pyplot(fig_dep)
            plt.close(fig_dep)
        except Exception as e:
            st.error(f"Error generating SHAP Dependence Plot: {e}")
            st.info("Ensure the selected feature and interaction feature exist and are numeric.")
    else:
        st.info("Not enough features to generate a dependence plot.")

    st.subheader("Local Prediction Explanation (SHAP Force Plot)")
    st.write("Visualizes how features contribute to a single prediction, pushing the prediction higher (red) or lower (blue) than the base value.")

    # Re-define render_force_plot_html for clarity and correctness
    def render_force_plot_html(explainer_obj, shap_values_obj, feature_values_obj, feature_names_list, predicted_class_idx=None):
        """Renders a SHAP force plot as HTML in Streamlit."""
        shap.initjs() # Initialize SHAP's JavaScript for rendering

        if isinstance(shap_values_obj, list): # Multi-class SHAP values
            if predicted_class_idx is not None:
                # Use SHAP values for the predicted class
                force_plot = shap.force_plot(
                    explainer_obj.expected_value[predicted_class_idx],
                    shap_values_obj[predicted_class_idx],
                    feature_values_obj,
                    feature_names=feature_names_list,
                    matplotlib=False
                )
            else:
                st.warning("Cannot generate force plot for multi-class without a specific class index.")
                return
        else: # Binary or regression SHAP values
            force_plot = shap.force_plot(
                explainer_obj.expected_value,
                shap_values_obj,
                feature_values_obj,
                feature_names=feature_names_list,
                matplotlib=False
            )

        # Capture HTML output
        shap_html = io.StringIO()
        shap.save_html(shap_html, force_plot)
        components.html(shap_html.getvalue(), height=350, scrolling=True)

    idx = st.slider("Select Sample Index for Force Plot", 0, len(X_test_scaled_df) - 1, 0)
    
    try:
        selected_instance_data = X_test_scaled_df.iloc[idx]
        predicted_class_for_instance = model.predict(selected_instance_data.to_frame().T)[0]

        st.write(f"**Explaining prediction for sample index:** `{idx}`")
        st.write(f"**Actual class:** `{y_test.iloc[idx]}`")
        st.write(f"**Predicted class:** `{predicted_class_for_instance}`")

        # Pass the specific SHAP explanation for the selected instance
        # If shap_values is a list (multi-class), select the explanation for the predicted class
        if isinstance(shap_values, list):
            render_force_plot_html(explainer, shap_values[predicted_class_for_instance][idx], selected_instance_data, features, predicted_class_for_instance)
        else: # Binary or regression
            render_force_plot_html(explainer, shap_values[idx], selected_instance_data, features)

    except Exception as e:
        st.error(f"Error generating SHAP Force Plot: {e}")
        st.info("Ensure the selected sample index is valid and the model/SHAP explainer are correctly set up.")

    st.markdown("---")
    st.header("📎 ELI5 Model Explanation (Permutation Importance)")
    st.write("ELI5 Permutation Importance measures how much the model's prediction error increases when a feature's values are randomly shuffled. This indicates the feature's importance.")

    try:
        # Use X_test_scaled_df for ELI5
        perm = eli5.sklearn.PermutationImportance(model, random_state=42).fit(X_test_scaled_df, y_test)
        
        # Ensure target_names are passed for multi-class classification if available
        eli5_html = eli5.show_weights(perm, feature_names=features, target_names=[str(c) for c in np.unique(y)])._repr_html_()
        components.html(eli5_html, height=500, scrolling=True)
    except Exception as e:
        st.error(f"Error generating ELI5 Permutation Importance: {e}")
        st.info("Permutation Importance requires a fitted model and test data. Check if your model is properly trained.")

    if hasattr(model, "feature_importances_") and model_choice not in ["LogisticRegression"]: # Logistic Regression doesn't have feature_importances_ directly
        st.subheader("📌 Model's Native Feature Importances")
        st.write("This section shows feature importances as reported directly by tree-based models (e.g., Random Forest, XGBoost, LightGBM).")
        try:
            importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            importances = importances.sort_values(by="Importance", ascending=False).reset_index(drop=True)
            
            fig_fi, ax_fi = plt.subplots(figsize=(10, len(features) * 0.5 + 2)) # Dynamic height
            sns.barplot(x="Importance", y="Feature", data=importances, ax=ax_fi)
            ax_fi.set_title(f"Feature Importances from {model_choice}")
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        except Exception as e:
            st.error(f"Error displaying native feature importances: {e}")
    else:
        st.info(f"The selected model ({model_choice}) does not directly expose `feature_importances_` attribute.")

    st.success("✅ Model interpretation complete!")
else:
    st.warning("📤 Upload a dataset (CSV format) to begin. Ensure it has at least two columns.")

