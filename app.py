import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

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
    # For tree models, TreeExplainer is more efficient. For others, use Explainer.
    try:
        explainer = shap.TreeExplainer(model)
        # For tree models, explainer.shap_values can sometimes return a list of arrays for multi-output
        # We need to handle this for classification models if they return shap values per class.
        # For binary classification, we often focus on shap_values[1] (for the positive class).
        shap_values = explainer.shap_values(X_data_df)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Assuming binary classification, taking SHAP values for the positive class (index 1)
            # You might need to adjust this based on your specific model's output
            shap_values = shap_values[1]
    except Exception:
        # Fallback to general Explainer for non-tree models or if TreeExplainer fails
        explainer = shap.Explainer(model, X_data_df)
        shap_values = explainer(X_data_df)
        # If explainer(X_data_df) returns a shap.Explanation object, get its values
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values


    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the SHAP summary bar plot
    shap.summary_plot(
        shap_values,
        X_data_df, # Pass the DataFrame here for proper feature naming
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

# --- Main Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="ML Model Interpretation with SHAP")

    st.title("Machine Learning Model Interpretation with SHAP")
    st.write("This application demonstrates global and local interpretations of a sample classification model using SHAP (SHapley Additive exPlanations).")

    st.sidebar.header("Model and Data Settings")

    # --- Generate Sample Data ---
    num_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
    num_features = st.sidebar.slider("Number of features", 5, 20, 10)
    random_state = st.sidebar.slider("Random State for Data Generation", 0, 100, 42)

    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=int(num_features * 0.7), # About 70% informative features
        n_redundant=int(num_features * 0.2),   # About 20% redundant features
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Convert to DataFrame for better feature naming in SHAP plots
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")

    st.sidebar.markdown("---")
    st.sidebar.header("Model Training")
    if st.sidebar.button("Train Random Forest Model"):
        st.session_state['model_trained'] = True
        st.session_state['X_df'] = X_df
        st.session_state['y_series'] = y_series

        with st.spinner("Training model..."):
            # Split data (though SHAP typically uses training data for background distribution)
            # For simplicity, we'll use the whole X_df as X_train for SHAP here
            # In a real scenario, you'd use X_train for explainer background and X_test for predictions
            # X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=random_state)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            clf.fit(X_df, y_series)
            st.session_state['model'] = clf
        st.success("Model trained successfully!")
    else:
        if 'model_trained' not in st.session_state:
            st.session_state['model_trained'] = False
            st.warning("Click 'Train Random Forest Model' in the sidebar to proceed.")


    if st.session_state.get('model_trained', False):
        model = st.session_state['model']
        X_data_for_shap = st.session_state['X_df']

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
                0 # Default to the first instance
            )

            # Ensure the explainer is available or re-create it
            if 'explainer' not in st.session_state:
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = shap.Explainer(model, X_data_for_shap)
                st.session_state['explainer'] = explainer

            explainer = st.session_state['explainer']
            single_instance = X_data_for_shap.iloc[[instance_index]]

            # Calculate SHAP values for the selected instance
            shap_values_instance = explainer(single_instance)

            st.write(f"**Explanation for Instance {instance_index}:**")
            st.dataframe(single_instance)

            fig_ind, ax_ind = plt.subplots(figsize=(10, 6))

            if isinstance(shap_values_instance, list) and len(shap_values_instance) > 1:
                # For classification, plot the explanation for the predicted class
                predicted_class = model.predict(single_instance)[0]
                st.write(f"Predicted Class: **{predicted_class}**")
                # Ensure we are using the Explanation object directly from explainer(instance) call
                # and selecting the correct output index for the predicted class
                shap.plots.waterfall(
                    shap_values_instance[predicted_class][0], # [0] because it's a single instance
                    show=False,
                    ax=ax_ind
                )
            elif hasattr(shap_values_instance, 'values'): # If it's a shap.Explanation object
                st.write(f"Predicted Value (Raw): **{model.predict(single_instance)[0]}**")
                shap.plots.waterfall(
                    shap_values_instance[0], # [0] because it's a single instance
                    show=False,
                    ax=ax_ind
                )
            else: # Fallback for other shap_values formats
                st.warning("Could not determine appropriate SHAP plot for this instance's SHAP values format.")
                # You might need to add more specific handling based on your model's output
                # For general cases, you might try force_plot or other plots
                pass


            ax_ind.set_title(f"SHAP Waterfall Plot for Instance {instance_index}")
            plt.tight_layout()
            st.pyplot(fig_ind)

            st.markdown("""
            ---
            **How to interpret the Waterfall Plot:**
            * **Base Value (E[f(X)])**: The average model output over the training dataset.
            * **Features (e.g., Feature_X = Value)**: Each bar shows the contribution of that feature's value to push the prediction from the base value to the final output.
            * **Red Bars**: Push the prediction higher (increase the predicted probability for the positive class).
            * **Blue Bars**: Push the prediction lower (decrease the predicted probability for the positive class).
            * **f(x)**: The final model output for this specific instance.
            """)
        else:
            st.info("No data available for local interpretation. Please train a model first.")

if __name__ == "__main__":
    main()
