import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np # Import numpy

# --- Function for Global SHAP Interpretation ---
def show_global_interpretation_shap(X_data, model):
    """
    Generates and displays a SHAP summary bar plot for global feature importance.

    Args:
        X_data (pd.DataFrame or np.array): The dataset used for calculating SHAP values.
                                           If a DataFrame, column names will be used as feature names.
        model: The trained machine learning model.
    """
    st.subheader("Global Feature Importance (SHAP Summary Plot - Bar)")

    # Ensure X_data is a DataFrame for better feature name display
    if not isinstance(X_data, pd.DataFrame):
        X_data_df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(X_data.shape[1])])
    else:
        X_data_df = X_data

    # Initialize SHAP explainer
    # Use try-except to handle different explainer types and their output formats
    try:
        explainer = shap.TreeExplainer(model)
        # For TreeExplainer, shap_values might be a list (for multi-output) or a single array
        raw_shap_values = explainer.shap_values(X_data_df)

        if isinstance(raw_shap_values, list) and len(raw_shap_values) > 1:
            # For classification, often focus on the SHAP values for the positive class (index 1)
            # Adjust this index if your positive class is at a different index or for multi-class
            shap_values_to_plot = raw_shap_values[1]
        else:
            shap_values_to_plot = raw_shap_values

    except Exception:
        # Fallback to general Explainer for non-tree models or if TreeExplainer fails
        explainer = shap.Explainer(model, X_data_df)
        shap_explanation = explainer(X_data_df) # This returns a shap.Explanation object
        shap_values_to_plot = shap_explanation.values # Extract the values array

    # Create a Matplotlib figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the SHAP summary bar plot
    # Pass feature_names if X_data_df is not used directly in shap_values
    shap.summary_plot(
        shap_values_to_plot,
        features=X_data_df, # Pass the DataFrame here for proper feature naming
        feature_names=X_data_df.columns.tolist(), # Explicitly pass feature names
        plot_type="bar",
        max_display=10, # Display top 10 features
        show=False,     # Don't show the plot immediately (Streamlit handles it)
        color_bar=False, # No color bar for bar plot
        plot_size="auto", # Use "auto" or None for default sizing
        ax=ax           # Pass the axes object
    )

    # Clean up plot aesthetics if needed
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

# --- Function for Local SHAP Interpretation ---
def show_local_interpretation_shap(X_data, model, explainer, instance_index):
    st.subheader(f"Local Feature Importance (SHAP Waterfall Plot for Instance {instance_index})")

    if X_data.empty:
        st.warning("No data available for local interpretation.")
        return

    single_instance = X_data.iloc[[instance_index]]
    
    # Calculate SHAP values for the selected instance
    # This will return a shap.Explanation object or a list of Explanation objects
    shap_explanation_instance = explainer(single_instance)

    st.write(f"**Selected Instance Data:**")
    st.dataframe(single_instance)

    # Determine the predicted class/value for display
    try:
        prediction = model.predict(single_instance)[0]
        if hasattr(model, 'predict_proba'): # For classification models
            probabilities = model.predict_proba(single_instance)[0]
            st.write(f"Predicted Class: **{prediction}** (Probability: {probabilities[prediction]:.2f})")
        else: # For regression models
            st.write(f"Predicted Value: **{prediction:.2f}**")
    except Exception as e:
        st.warning(f"Could not get model prediction: {e}")
        prediction = None # Set to None if prediction fails

    # SHAP waterfall plot often creates its own figure, so we might not need to pass an 'ax'
    # It's better to let shap.plots.waterfall manage its figure directly
    # and then capture it via plt.gcf() if you need to pass it to st.pyplot.

    if isinstance(shap_explanation_instance, list) and len(shap_explanation_instance) > 1:
        # For multi-output (e.g., multi-class classification), select the explanation for the predicted class
        if prediction is not None:
            # Access the Explanation object for the predicted class
            explanation_for_plot = shap_explanation_instance[prediction]
            fig_ind = shap.plots.waterfall(explanation_for_plot, show=False)
        else:
            st.error("Cannot plot waterfall without a valid prediction for multi-output model.")
            return # Exit if prediction failed for multi-output
    elif isinstance(shap_explanation_instance, shap.Explanation):
        # For single output or if explainer returns a single Explanation object
        # Ensure we're passing a single Explanation object (e.g., the first row if multiple instances were passed)
        fig_ind = shap.plots.waterfall(shap_explanation_instance[0], show=False)
    else:
        st.error("Unsupported SHAP explanation format for waterfall plot.")
        return

    if fig_ind: # Only proceed if a figure was generated
        plt.tight_layout()
        st.pyplot(fig_ind)
        plt.close(fig_ind) # Close the figure to free up memory

    st.markdown("""
    ---
    **How to interpret the Waterfall Plot:**
    * **Base Value (E[f(X)])**: The average model output over the background dataset.
    * **Features (e.g., Feature_X = Value)**: Each bar shows the contribution of that feature's value to push the prediction from the base value to the final output.
    * **Red Bars**: Indicate features that push the prediction higher.
    * **Blue Bars**: Indicate features that push the prediction lower.
    * **f(x)**: The final model output for this specific instance.
    """)


# --- Main Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="ML Model Interpretation with SHAP")

    st.title("Machine Learning Model Interpretation with SHAP")
    st.write("This application demonstrates global and local interpretations of a sample classification model using SHAP (SHapley Additive exPlanations).")

    st.sidebar.header("Model and Data Settings")

    # --- Generate Sample Data ---
    num_samples = st.sidebar.slider("Number of samples", 100, 1000, 500, key='num_samples_slider')
    num_features = st.sidebar.slider("Number of features", 5, 20, 10, key='num_features_slider')
    random_state = st.sidebar.slider("Random State for Data Generation", 0, 100, 42, key='random_state_slider')

    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=int(num_features * 0.7),
        n_redundant=int(num_features * 0.2),
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Convert to DataFrame for better feature naming in SHAP plots
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")

    st.sidebar.markdown("---")
    st.sidebar.header("Model Training")
    
    # Initialize session state for model_trained if not present
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
        st.session_state['model'] = None
        st.session_state['X_df'] = pd.DataFrame()
        st.session_state['y_series'] = pd.Series()
        st.session_state['explainer'] = None # Store explainer

    if st.sidebar.button("Train Random Forest Model", key='train_button'):
        st.session_state['model_trained'] = True
        st.session_state['X_df'] = X_df
        st.session_state['y_series'] = y_series

        with st.spinner("Training model..."):
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            clf.fit(X_df, y_series)
            st.session_state['model'] = clf

            # Re-initialize explainer after model training
            try:
                st.session_state['explainer'] = shap.TreeExplainer(clf)
            except Exception:
                st.session_state['explainer'] = shap.Explainer(clf, X_df) # Pass X_df as background
        st.success("Model trained successfully!")
    else:
        if not st.session_state['model_trained']:
            st.warning("Click 'Train Random Forest Model' in the sidebar to proceed.")


    if st.session_state.get('model_trained', False) and st.session_state['model'] is not None:
        model = st.session_state['model']
        X_data_for_shap = st.session_state['X_df']
        explainer = st.session_state['explainer']

        st.markdown("---")
        st.header("1. Global Interpretation")
        show_global_interpretation_shap(X_data_for_shap, model)

        st.markdown("---")
        st.header("2. Local Interpretation (Individual Prediction Explanation)")
        st.write("Select an instance from the dataset to see its individual SHAP explanation.")

        if not X_data_for_shap.empty:
            instance_index = st.slider(
                "Select data instance index",
                0,
                len(X_data_for_shap) - 1,
                0, # Default to the first instance
                key='instance_slider'
            )
            
            show_local_interpretation_shap(X_data_for_shap, model, explainer, instance_index)
        else:
            st.info("No data available for local interpretation. Please train a model first.")

if __name__ == "__main__":
    main()
