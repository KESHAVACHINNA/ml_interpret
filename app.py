import streamlit as st
import pandas as pd
import numpy as np

# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
# DMatrix is no longer needed for prediction with the wrapper
# from xgboost import DMatrix 

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

def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
        df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox("Then choose the target variable", col_arranged)
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "iris":
        df = sns.load_dataset("iris")
        target_col = "species"
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "titanic":
        df = sns.load_dataset("titanic").drop(columns=["class", "who", "adult_male", "deck", "alive", "alone"])
        target_col = "survived"
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "census income":
        X, y = shap.datasets.adult()
        features = X.columns
        target_labels = pd.Series(y).unique()
        df = pd.concat([X, pd.DataFrame(y, columns=["Outcome"])], axis=1)
    return df, X, y, features, target_labels

def encode_data(data, targetcol):
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    features = X.columns
    data[targetcol] = data[targetcol].astype("object")
    target_labels = pd.Series(data[targetcol]).unique()
    y = pd.factorize(data[targetcol])[0]
    return X, y, features, target_labels

def splitdata(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)
    return X_train, X_test, y_train, y_test

# CORRECTED FUNCTION
def make_pred(dim_model, X_test, clf):
    # No special case needed for XGBoost anymore, as the wrapper has a .predict method
    pred = clf.predict(X_test)
    return pred

def show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    # This function should now work correctly with XGBoost because we are using the scikit-learn wrapper
    if dim_model == "XGBoost":
        # Using PermutationImportance for consistency and robustness across all models
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(X_train, y_train)
        df_global_explain = eli5.explain_weights_df(perm, feature_names=features.tolist(), top=5).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(X_train, y_train)
        df_global_explain = eli5.explain_weights_df(perm, feature_names=features.tolist(), top=5).round(2)
        
    bar = (alt.Chart(df_global_explain).mark_bar(color="red", opacity=0.6, size=16)
           .encode(x="weight", y=alt.Y("feature", sort="-x"), tooltip=["weight"]).properties(height=160))
    st.write(bar)

def show_global_interpretation_shap(X_train, clf):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    fig, ax = plt.subplots(figsize=(12, 5))
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=5, 
                     color=plt.get_cmap("tab20b"), show=False, color_bar=False, ax=ax)
    st.pyplot(fig)


def filter_misclassified(X_test, y_test, pred):
    idx_misclassified = pred != y_test
    X_test_misclassified = X_test[idx_misclassified]
    y_test_misclassified = y_test[idx_misclassified]
    pred_misclassified = pred[idx_misclassified]
    return X_test_misclassified, y_test_misclassified, pred_misclassified

def show_local_interpretation_eli5(dataset, clf, pred, target_labels, features, dim_model, slider_idx):
    info_local = st.button("How this works")
    if info_local:
        st.info("""**What's included** Input data is split 80/20 into training and testing. <br>
        Each of the individual testing datapoint can be inspected by index.<br>
        **To Read the table** The table describes how an individual datapoint is classified.<br>
        Contribution refers to the extent & direction of influence a feature has on the outcome<br>
        Value refers to the value of the feature in the dataset. Bias means an intercept.""")

    # This function should now work correctly with XGBoost because we are using the scikit-learn wrapper
    local_interpretation = eli5.show_prediction(clf, doc=dataset.iloc[slider_idx, :],
                                               target_names=target_labels.tolist(), 
                                               show_feature_values=True, top=5, targets=[True])
    st.markdown(local_interpretation.data.replace("\n", ""), unsafe_allow_html=True)

def show_local_interpretation_shap(clf, X_test, pred, target_labels, slider_idx):
    info_local = st.button("How this works")
    if info_local:
        st.info("""This waterfall plot shows how the model's output changes from the expected value 
        (base value) to the final predicted value for a single instance. Each bar represents the impact 
        of a feature, with red bars increasing the prediction and blue bars decreasing it.""")
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    pred_i = int(pred[slider_idx])

    if isinstance(shap_values, list): # For multi-class classification
        shap_values_for_plot = shap_values[pred_i][slider_idx]
        expected_value_for_plot = explainer.expected_value[pred_i]
    else: # For binary classification
        shap_values_for_plot = shap_values[slider_idx]
        expected_value_for_plot = explainer.expected_value

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=shap_values_for_plot, base_values=expected_value_for_plot,
                                         data=X_test.iloc[slider_idx, :].values, 
                                         feature_names=X_test.columns.tolist()), show=False, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

def show_local_interpretation(X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework):
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data - 1)
    st.text("Prediction: " + str(target_labels[int(pred[slider_idx])]) + 
            " | Actual label: " + str(target_labels[int(y_test[slider_idx])]))

    if dim_framework == "SHAP":
        show_local_interpretation_shap(clf, X_test, pred, target_labels, slider_idx)
    elif dim_framework == "ELI5":
        show_local_interpretation_eli5(X_test, clf, pred, target_labels, features, dim_model, slider_idx)

def show_perf_metrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
    fig, ax = plt.subplots()
    conf_matrix = confusion_matrix(y_test, pred, labels=list(set(y_test)))
    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix, square=True, annot=True, annot_kws={"size": 15}, 
                cmap="YlGnBu", cbar=False, ax=ax)
    st.sidebar.pyplot(fig)

def draw_pdp(clf, dataset, features, target_labels, dim_model):
    # This condition is no longer strictly necessary but can be kept for consistency
    if dim_model != "XGBoost_native": # Or just remove the condition entirely
        selected_col = st.selectbox("Select a feature", features)
        st.info("""**To read the chart:** The curves describe how a feature marginally varies with 
        the likelihood of outcome. Each subplot belong to a class outcome. When a curve is below 0, 
        the data is unlikely to belong to that class.""")

        pdp_dist = pdp.pdp_isolate(model=clf, dataset=dataset, model_features=features, feature=selected_col)
        ncol = len(target_labels) if len(target_labels) <= 5 else 5
        fig, axes = plt.subplots(ncols=ncol, figsize=(12, 5))
        pdp.pdp_plot(pdp_dist, selected_col, plot_pts_vert=False, show_titles=True, ax=axes)
        st.pyplot(fig)

def main():
    dim_data = st.sidebar.selectbox("Try out sample data", ("iris", "titanic", "census income"))
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")
    df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)
    X_train, X_test, y_train, y_test = splitdata(X, y)

    dim_model = st.sidebar.selectbox("Choose a model", ("XGBoost", "lightGBM", "randomforest"))
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "lightGBM":
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(class_weight="balanced", objective="multiclass", n_jobs=-1, verbose=-1)
        else:
            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
    
    # CORRECTED BLOCK
    elif dim_model == "XGBoost":
        # Use the scikit-learn wrapper: XGBClassifier for ELI5 compatibility
        clf = xgb.XGBClassifier(use_label_encoder=False, 
                                eval_metric='mlogloss', 
                                random_state=2)
        clf.fit(X_train, y_train)

    pred = make_pred(dim_model, X_test, clf)
    dim_framework = st.sidebar.radio("Choose interpretation framework", ["SHAP", "ELI5"])

    if st.sidebar.checkbox("Preview uploaded data"):
        st.sidebar.dataframe(df.head())

    st.sidebar.markdown("#### Classification report")
    show_perf_metrics(y_test, pred)

    st.markdown("#### Global Interpretation")
    st.text("Most important features")
    info_global = st.button("How it is calculated")
    if info_global:
        st.info("""The importance of each feature is derived from permutation importance - 
        by randomly shuffle a feature, how much does the model performance decrease.""")
    
    if dim_framework == "SHAP":
        show_global_interpretation_shap(X_train, clf)
    elif dim_framework == "ELI5":
        show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model)

    st.markdown("#### Local Interpretation")
    if st.checkbox("Filter for misclassified"):
        X_test, y_test, pred = filter_misclassified(X_test, y_test, pred)
        if X_test.shape[0] == 0:
            st.text("No misclassification🎉")
        else:
            st.text(str(X_test.shape[0]) + " misclassified total")
            show_local_interpretation(X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework)
    else:
        show_local_interpretation(X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework)

    # The PDP plot condition can be simplified since all models now have the same interface
    if st.checkbox("Show how features vary with outcome"):
        draw_pdp(clf, X_train, features, target_labels, dim_model)

if __name__ == "__main__":
    main()
