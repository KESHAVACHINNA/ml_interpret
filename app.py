# app.py

import streamlit as st
import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Explainability
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import shap
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("ML Interpreter 🔍")
st.subheader("Visual Explanations of Black Box Models")


@st.cache_data
def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in df.columns]
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox("Select target column", col_arranged)
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
    y = pd.factorize(data[targetcol])[0]
    target_labels = data[targetcol].astype(str).unique()
    return X, y, features, target_labels


def split_data(X, y):
    return train_test_split(X, y, train_size=0.8, random_state=0)


def make_prediction(dim_model, X_test, clf):
    if dim_model == "XGBoost":
        return clf.predict(DMatrix(X_test))
    return clf.predict(X_test)


def global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    if dim_model == "XGBoost":
        weights_df = eli5.explain_weights_df(clf, feature_names=features.values, top=5).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(X_train, y_train)
        weights_df = eli5.explain_weights_df(perm, feature_names=features.values, top=5).round(2)
    chart = alt.Chart(weights_df).mark_bar().encode(
        x="weight:Q",
        y=alt.Y("feature:N", sort='-x'),
        tooltip=["weight"]
    ).properties(title="Top Features")
    st.altair_chart(chart, use_container_width=True)


def global_interpretation_shap(X_train, clf):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')


def show_local_explanation_eli5(X_test, clf, target_labels, features, dim_model, idx):
    if dim_model == "XGBoost":
        html = eli5.show_prediction(clf, doc=X_test.iloc[idx], show_feature_values=True, top=5)
    else:
        html = eli5.show_prediction(clf, doc=X_test.iloc[idx], feature_names=features.values, top=5)
    st.markdown(html.data.replace("\n", ""), unsafe_allow_html=True)


def show_local_explanation_shap(X_test, clf, pred, idx):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    pred_class = int(pred[idx])
    shap.force_plot(
        explainer.expected_value[pred_class],
        shap_values[pred_class][idx],
        X_test.iloc[idx],
        matplotlib=True
    )
    st.pyplot(bbox_inches='tight')


def performance_metrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.markdown("### Classification Report")
    st.sidebar.dataframe(pd.DataFrame(report).T.round(2))

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    st.sidebar.pyplot(fig)


def main():
    st.sidebar.title("Options")

    dim_data = st.sidebar.selectbox("Select a sample dataset", ["iris", "titanic", "census income"])
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")

    df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model_choice = st.sidebar.selectbox("Choose a model", ["XGBoost", "lightGBM", "randomforest"])
    if model_choice == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
    elif model_choice == "lightGBM":
        clf = lgb.LGBMClassifier(n_jobs=-1).fit(X_train, y_train)
    else:
        dtrain = DMatrix(X_train, label=y_train)
        clf = xgb.train({'max_depth': 5, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class': len(np.unique(y))}, dtrain)

    pred = make_prediction(model_choice, X_test, clf)

    interpretation_tool = st.sidebar.radio("Interpretation framework", ["SHAP", "ELI5"])

    if st.sidebar.checkbox("Show preview of data"):
        st.write(df.head())

    performance_metrics(y_test, pred)

    st.markdown("### 🔍 Global Interpretation")
    if interpretation_tool == "SHAP":
        global_interpretation_shap(X_train, clf)
    else:
        global_interpretation_eli5(X_train, y_train, features, clf, model_choice)

    st.markdown("### 🎯 Local Interpretation")
    idx = st.slider("Select index", 0, len(X_test) - 1)
    st.write(f"Prediction: {target_labels[int(pred[idx])]}, Actual: {target_labels[int(y_test[idx])]}")
    if interpretation_tool == "SHAP":
        show_local_explanation_shap(X_test, clf, pred, idx)
    else:
        show_local_explanation_eli5(X_test, clf, target_labels, features, model_choice, idx)


if __name__ == "__main__":
    main()
