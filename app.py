import streamlit as st
import pandas as pd
import numpy as np

# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# interpretation
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import shap

# Title and Subheader
st.title("ML Interpreter")
st.subheader("Blackbox ML classifiers visually explained")

# --- Streamlit Session State Initialization ---
# Initialize session state variables if they don't exist to persist choices across reruns [1, 2]
if 'dim_data' not in st.session_state:
    st.session_state.dim_data = "iris"
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'dim_model' not in st.session_state:
    st.session_state.dim_model = "XGBoost"
if 'dim_framework' not in st.session_state:
    st.session_state.dim_framework = "SHAP"
if 'filter_misclassified_checkbox' not in st.session_state:
    st.session_state.filter_misclassified_checkbox = False

# --- Data Loading and Preprocessing Functions ---

@st.cache_data(show_spinner="Loading and preprocessing data...") # Cache data loading and initial processing [3]
def upload_data(uploaded_file, dim_data_choice):
    df = None
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
    elif dim_data_choice == "iris":
        df = sns.load_dataset("iris")
    elif dim_data_choice == "titanic":
        df = sns.load_dataset("titanic").drop(
            columns=["class", "who", "adult_male", "deck", "alive", "alone"]
        )
    elif dim_data_choice == "census income":
        X_shap, y_shap = shap.datasets.adult()
        df = pd.concat()], axis=1)

    if df is not None:
        # Robust column name sanitization [4]
        df.columns = df.columns.str.strip() # Remove leading/trailing whitespace
        df.columns = df.columns.str.lower().str.replace(" ", "_") # Lowercase and replace spaces with underscores
        df.columns = df.columns.str.replace("[^a-z0-9_]", "", regex=True) # Remove special characters [4]

    return df

@st.cache_data(show_spinner="Encoding data...") # Cache data encoding [3]
def encode_data(data, target_col):
    """Preprocess categorical and numerical values, handle missing values."""
    if data is None:
        return None, None, None, None

    # Separate features (X) and target (y)
    X_df = data.drop(target_col, axis=1)
    y_series = data[target_col]

    # Handle missing values and encode features
    numerical_cols = X_df.select_dtypes(include=np.number).columns
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns

    # Impute numerical columns [5, 6, 7]
    if not numerical_cols.empty:
        imputer_numerical = SimpleImputer(strategy='median')
        X_df[numerical_cols] = imputer_numerical.fit_transform(X_df[numerical_cols])

    # One-hot encode categorical columns, handling NaNs explicitly [8]
    if not categorical_cols.empty:
        X_encoded_categorical = pd.get_dummies(X_df[categorical_cols], dummy_na=True)
        # Drop original categorical columns and concatenate one-hot encoded ones
        X_df = pd.concat([X_df.drop(columns=categorical_cols), X_encoded_categorical], axis=1)

    # Ensure all column names are valid for LightGBM/XGBoost (already done by sanitization in upload_data)
    X_df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_df.columns]
    features = X_df.columns

    # Factorize target variable (y) [9]
    target_labels = y_series.unique()
    y = pd.factorize(y_series)

    return X_df, y, features, target_labels

@st.cache_data(show_spinner="Splitting data...") # Cache data splitting [3]
def splitdata(X, y):
    """Split dataset into training & testing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0
    )
    return X_train, X_test, y_train, y_test

@st.cache_resource(show_spinner="Training model...") # Cache model training [3, 10]
def train_model(X_train, y_train, dim_model, target_labels):
    """Train the selected machine learning model."""
    clf = None
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1) # Good default parameters [11, 12]
        clf.fit(X_train, y_train)
    elif dim_model == "lightGBM":
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(
                class_weight="balanced", objective="multiclass", n_jobs=-1, verbose=-1 # Correct objective and verbosity [13, 14]
            )
        else:
            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1) # Correct objective and verbosity [13, 14]
        clf.fit(X_train, y_train)
    elif dim_model == "XGBoost":
        params = {
            "max_depth": 5,
            "verbosity": 0, # Replaced 'silent: 1' with 'verbosity: 0' [15, 16]
            "random_state": 2,
            "num_class": len(target_labels),
            "objective": "multi:softprob" if len(target_labels) > 2 else "binary:logistic", # Explicit objective [17, 18]
        }
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix)
    return clf

@st.cache_data(show_spinner="Making predictions...") # Cache predictions [3]
def make_pred(dim_model, X_test, clf):
    """Get y_pred using the classifier"""
    if dim_model == "XGBoost":
        # For XGBoost Booster, predict returns raw scores or probabilities based on objective
        # For multi-class, it returns probabilities for each class, need argmax for class label
        if clf.params.get("objective") == "multi:softprob":
            pred_proba = clf.predict(DMatrix(X_test))
            pred = np.argmax(pred_proba, axis=1)
        else: # binary:logistic or other binary objectives
            pred = (clf.predict(DMatrix(X_test)) > 0.5).astype(int) # Convert probabilities to binary labels
    elif dim_model == "lightGBM":
        pred = clf.predict(X_test)
    else: # RandomForest
        pred = clf.predict(X_test)
    return pred

@st.cache_data(show_spinner="Filtering misclassified instances...") # Cache filtering [3]
def filter_misclassified(X_test, y_test, pred):
    """Get misclassified instances"""
    idx_misclassified = pred!= y_test
    X_test_misclassified = X_test[idx_misclassified]
    y_test_misclassified = y_test[idx_misclassified]
    pred_misclassified = pred[idx_misclassified]
    return X_test_misclassified, y_test_misclassified, pred_misclassified

# --- Interpretation Functions ---

def show_global_interpretation_eli5(X_test, y_test, features, clf, dim_model):
    """Show most important features via permutation importance in ELI5"""
    st.info(
        """
        Permutation Importance measures how much the model's score decreases when a feature's values are randomly shuffled.
        A larger decrease indicates higher importance. This is calculated on the test set to reflect generalization. [7, 19, 20]
        """
    )
    # Permutation importance should be calculated on unseen data (X_test, y_test) for generalization [7, 19, 20]
    perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(X_test, y_test)
    df_global_explain = eli5.explain_weights_df(
        perm, feature_names=features.values, top=5
    ).round(2)
    bar = (
        alt.Chart(df_global_explain)
       .mark_bar(color="red", opacity=0.6, size=16)
       .encode(x="weight", y=alt.Y("feature", sort="-x"), tooltip=["weight"])
       .properties(height=160)
    )
    st.write(bar)


def show_global_interpretation_shap(X_train, clf):
    """Show most important features via permutation importance in SHAP"""
    st.info(
        """
        SHAP (SHapley Additive exPlanations) values represent the contribution of each feature to the prediction.
        The summary plot shows the average absolute SHAP value for each feature, indicating overall feature importance.
        """
    )
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train) # SHAP values can be computed on training data for global insights

    # Handle multiclass shap_values for summary_plot [21]
    if isinstance(shap_values, list): # For multiclass, shap_values is a list of arrays
        # For a bar plot, we typically show importance for one class or an aggregated view.
        # Using the first class's SHAP values for the bar plot as a default.
        shap.summary_plot(
            shap_values,
            X_train,
            plot_type="bar",
            max_display=5,
            plot_size=(12, 5),
            color=plt.get_cmap("tab20b"),
            show=False,
            color_bar=False,
        )
    else: # Binary classification or regression
        shap.summary_plot(
            shap_values,
            X_train,
            plot_type="bar",
            max_display=5,
            plot_size=(12, 5),
            color=plt.get_cmap("tab20b"),
            show=False,
            color_bar=False,
        )
    st.pyplot()


def show_local_interpretation_eli5(
    dataset, clf, pred, target_labels, features, dim_model, slider_idx
):
    """Show the interpretation of individual decision points using ELI5"""
    st.info(
        """
        ELI5 explains individual predictions by showing the contribution of each feature to the predicted outcome.
        Positive weights (green) push the prediction towards the target class, while negative weights (red) push it away.
        """
    )

    # Ensure target_names is always passed for ELI5 for clarity, especially in multiclass [3, 22, 23, 24]
    # For binary classification, targets= often means explaining the positive class.
    # For multiclass, we explain the predicted class.
    eli5_targets = [target_labels[int(pred[slider_idx])]] if len(target_labels) > 2 else

    local_interpretation = eli5.show_prediction(
        clf,
        doc=dataset.iloc[slider_idx, :],
        show_feature_values=True,
        top=5,
        target_names=list(target_labels), # Pass target_names for all models [3, 22, 23, 24]
        targets=eli5_targets,
    )
    st.markdown(
        local_interpretation.data.replace("\n", ""), unsafe_allow_html=True,
    )


def show_local_interpretation_shap(clf, X_test, pred, slider_idx):
    """Show the interpretation of individual decision points using SHAP"""
    st.info(
        """
        This chart illustrates how each feature collectively influences the prediction outcome for a single instance.
        Features in red push the prediction higher (towards the predicted class), and features in blue push it lower.
        The base value is the average model output, and the SHAP values explain how each feature moves the prediction from this base value to the final output.
       (https://github.com/slundberg/shap)
        Please note that the explanation here is always based on the predicted class rather than the positive class (i.e. if predicted class is 0, to the right means more likely to be 0) to cater for multi-class scenarios.
        """
    )
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # The predicted class for the selected instance
    pred_i = int(pred[slider_idx])

    # Handle multiclass shap_values for force_plot [21, 25]
    if isinstance(shap_values, list): # Multiclass
        # shap_values is a list of arrays, one for each class
        # We need to select the SHAP values corresponding to the predicted class
        shap.force_plot(
            explainer.expected_value[pred_i],
            shap_values[pred_i][slider_idx, :],
            X_test.iloc[slider_idx, :],
            matplotlib=True,
            show=False # Ensure plot is not shown directly by shap
        )
    else: # Binary or regression
        shap.force_plot(
            explainer.expected_value, # For binary/regression, expected_value is a single value
            shap_values[slider_idx, :],
            X_test.iloc[slider_idx, :],
            matplotlib=True,
            show=False # Ensure plot is not shown directly by shap
        )
    st.pyplot()


def show_local_interpretation(
    X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework
):
    """Show the interpretation based on the selected framework"""
    n_data = X_test.shape
    if n_data == 0:
        st.text("No data points to explain.")
        return

    slider_idx = st.slider("Which datapoint to explain", 0, n_data - 1)

    st.text(
        "Prediction: "
        + str(target_labels[int(pred[slider_idx])])
        + " | Actual label: "
        + str(target_labels[int(y_test[slider_idx])])
    )

    if dim_framework == "SHAP":
        show_local_interpretation_shap(clf, X_test, pred, slider_idx)
    elif dim_framework == "ELI5":
        show_local_interpretation_eli5(
            X_test, clf, pred, target_labels, features, dim_model, slider_idx
        )


def show_perf_metrics(y_test, pred):
    """Show model performance metrics such as classification report or confusion matrix"""
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
    
    # Ensure labels for confusion matrix are correct
    unique_y_test = list(np.unique(y_test))
    unique_pred = list(np.unique(pred))
    all_labels = sorted(list(set(unique_y_test + unique_pred)))

    conf_matrix = confusion_matrix(y_test, pred, labels=all_labels)
    
    fig, ax = plt.subplots() # Create a figure and an axes object for the plot
    sns.set(font_scale=1.4)
    sns.heatmap(
        conf_matrix,
        square=True,
        annot=True,
        annot_kws={"size": 15},
        cmap="YlGnBu",
        cbar=False,
        fmt='d', # Format annotations as integers
        xticklabels=all_labels, # Set x-axis labels
        yticklabels=all_labels, # Set y-axis labels
        ax=ax # Pass the axes object to seaborn
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.sidebar.pyplot(fig) # Pass the figure object to Streamlit


def draw_pdp(clf, dataset, features, target_labels, dim_model):
    """Draw pdpplot given a model, data, all the features and the selected feature to plot"""

    selected_col = st.selectbox("Select a feature", features)
    st.info(
        """
        Partial Dependence Plots (PDPs) show the marginal effect of one or two features on the predicted outcome of a machine learning model.
        They illustrate how the prediction changes on average as the feature(s) vary.
        **Note:** PDPs assume that the feature(s) for which the partial dependence is computed are independent of other features. [26]
        If features are highly correlated, the interpretation of a PDP might be misleading as it shows the effect of unlikely feature combinations.
        """
    )

    # Define a prediction function for XGBoost Booster for PDPbox compatibility [27]
    def xgb_predict_proba(model, X):
        # For multi-class, xgb.train with multi:softprob returns probabilities directly
        # For binary, xgb.train with binary:logistic returns probabilities
        # PDPbox expects predict_proba to return (n_samples, n_classes) array
        if model.params.get("objective") == "multi:softprob":
            return model.predict(DMatrix(X))
        elif model.params.get("objective") == "binary:logistic":
            # For binary, it returns a 1D array of probabilities for the positive class
            # Reshape to (n_samples, 2) for PDPbox (probability of negative class, probability of positive class)
            preds = model.predict(DMatrix(X))
            return np.vstack([1 - preds, preds]).T
        else:
            # Fallback for other objectives, might need adjustment
            return model.predict(DMatrix(X))

    # Use custom pred_func for XGBoost Booster, otherwise use default
    pred_func = xgb_predict_proba if dim_model == "XGBoost" else None

    pdp_dist = pdp.pdp_isolate(
        model=clf, dataset=dataset, model_features=features, feature=selected_col,
        pred_func=pred_func # Pass custom prediction function if needed
    )
    if len(target_labels) <= 5:
        ncol = len(target_labels)
    else:
        ncol = 5
    
    fig, axes = pdp.pdp_plot(pdp_dist, selected_col, ncols=ncol, figsize=(12, 5)) # Get figure and axes
    st.pyplot(fig) # Pass figure to streamlit


def main():
    # Use session state for selectbox values to persist choices [1, 2]
    st.session_state.dim_data = st.sidebar.selectbox(
        "Try out sample data", ("iris", "titanic", "census income"),
        key="dim_data_selectbox",
        on_change=lambda: st.session_state.update(target_col=None) # Reset target_col when data changes
    )
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")

    df = upload_data(uploaded_file, st.session_state.dim_data)

    if df is None:
        st.warning("Please upload a CSV file or select a sample dataset.")
        return

    # Dynamically set target_col based on selected data
    if st.session_state.dim_data == "iris":
        default_target_col = "species"
    elif st.session_state.dim_data == "titanic":
        default_target_col = "survived"
    elif st.session_state.dim_data == "census income":
        default_target_col = "outcome" # Sanitized column name
    else: # For uploaded file, default to last column
        default_target_col = df.columns[-1]

    # Ensure target_col is in df.columns before setting it
    if st.session_state.target_col not in df.columns:
        st.session_state.target_col = default_target_col

    st.session_state.target_col = st.sidebar.selectbox(
        "Then choose the target variable", df.columns,
        index=list(df.columns).index(st.session_state.target_col) if st.session_state.target_col in df.columns else 0,
        key="target_col_selectbox"
    )

    X, y, features, target_labels = encode_data(df, st.session_state.target_col)

    if X is None or y is None:
        st.error("Data encoding failed. Please check your dataset or target column selection.")
        return

    X_train, X_test, y_train, y_test = splitdata(X, y)
    
    st.session_state.dim_model = st.sidebar.selectbox(
        "Choose a model", ("XGBoost", "lightGBM", "randomforest"),
        key="dim_model_selectbox"
    )

    clf = train_model(X_train, y_train, st.session_state.dim_model, target_labels)
    pred = make_pred(st.session_state.dim_model, X_test, clf)

    st.session_state.dim_framework = st.sidebar.radio(
        "Choose interpretation framework",,
        key="dim_framework_radio"
    )

    if st.sidebar.checkbox("Preview uploaded data"):
        st.sidebar.dataframe(df.head())
    
    st.sidebar.markdown("#### Classification report")
    show_perf_metrics(y_test, pred)
    
    st.markdown("#### Global Interpretation")
    st.text("Most important features")
    
    # The info text is now part of the respective global interpretation functions
    # No need for a separate info button here.
    
    if st.session_state.dim_framework == "SHAP":
        show_global_interpretation_shap(X_train, clf)
    elif st.session_state.dim_framework == "ELI5":
        show_global_interpretation_eli5(X_test, y_test, features, clf, st.session_state.dim_model) # Pass X_test, y_test

    if st.sidebar.button("About the app"):
        st.sidebar.markdown(
            """
            This application provides an interactive platform to understand your Machine Learning models.
            Upload your data, select a model, and explore global and local interpretations using SHAP and ELI5.
            """
        )
        st.sidebar.markdown(
            '<a href="https://ctt.ac/zu8S4"><img src="https://image.flaticon.com/icons/svg/733/733579.svg" width=16></a>',
            unsafe_allow_html=True,
        )
    
    st.markdown("#### Local Interpretation")

    # Use session state for checkbox
    st.session_state.filter_misclassified_checkbox = st.checkbox(
        "Filter for misclassified",
        key="filter_misclassified_checkbox"
    )

    if st.session_state.filter_misclassified_checkbox:
        X_test_filtered, y_test_filtered, pred_filtered = filter_misclassified(X_test, y_test, pred)
        if X_test_filtered.shape == 0:
            st.text("No misclassification🎉")
        else:
            st.text(str(X_test_filtered.shape) + " misclassified total")
            show_local_interpretation(
                X_test_filtered,
                y_test_filtered,
                clf,
                pred_filtered,
                target_labels,
                features,
                st.session_state.dim_model,
                st.session_state.dim_framework,
            )
    else:
        show_local_interpretation(
            X_test, y_test, clf, pred, target_labels, features, st.session_state.dim_model, st.session_state.dim_framework
        )
    
    # PDPbox is now compatible with XGBoost Booster via custom pred_func
    if st.checkbox("Show how features vary with outcome (Partial Dependence Plots)"):
        draw_pdp(clf, X_train, features, target_labels, st.session_state.dim_model)


if __name__ == "__main__":
    main()
