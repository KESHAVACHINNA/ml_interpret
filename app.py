import streamlit as st
import pandas as pd

# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
st.subheader("Blackblox ML classifiers visually explained")


def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
        # replace all non alphanumeric column names to avoid lgbm issue
        df.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
        ]
        # make the last col the default outcome
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox(
            "Then choose the target variable", col_arranged
        )
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "iris":
        df = sns.load_dataset("iris")
        target_col = "species"
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "titanic":
        df = sns.load_dataset("titanic").drop(
            columns=["class", "who", "adult_male", "deck", "alive", "alone"]
        )
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
    target_labels = data[targetcol].unique()
    y = pd.factorize(data[targetcol])[0]
    return X, y, features, target_labels


def splitdata(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0
    )
    return X_train, X_test, y_train, y_test


def make_pred(dim_model, X_test, clf):
    if dim_model == "XGBoost":
        pred = clf.predict(DMatrix(X_test))
    elif dim_model == "lightGBM":
        pred = clf.predict(X_test)
    else:
        pred = clf.predict(X_test)
    return pred


def show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    if dim_model == "XGBoost":
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=features.values, top=5
        ).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(
            X_train, y_train
        )
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


import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def show_perf_metrics(y_true, y_pred):
    st.subheader("Performance Metrics")

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    st.text("Classification Report:")
    st.json(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)


def filter_misclassified(X_test, y_test, pred):
    """get misclassified instances"""
    idx_misclassified = pred != y_test
    X_test_misclassified = X_test[idx_misclassified]
    y_test_misclassified = y_test[idx_misclassified]
    pred_misclassified = pred[idx_misclassified]
    return X_test_misclassified, y_test_misclassified, pred_misclassified


def show_local_interpretation_eli5(
    dataset, clf, pred, target_labels, features, dim_model, slider_idx
):
    info_local = st.button("How this works")
    if info_local:
        st.info("ELI5 shows the contribution of each feature to a specific prediction. Positive weights indicate features that push the prediction towards the positive class, while negative weights push it towards the negative class.")

    if dim_model == "XGBoost":
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], show_feature_values=True, top=5
        )
    else:
        local_interpretation = eli5.show_prediction(
            clf,
            doc=dataset.iloc[slider_idx, :],
            target_names=target_labels,
            show_feature_values=True,
            top=5,
            targets=[True],
        )
    st.markdown(
        local_interpretation.data.replace("\n", ""), unsafe_allow_html=True,
    )


def show_local_interpretation_shap(clf, X_test, pred, slider_idx):
    info_local = st.button("How this works")
    if info_local:
        st.info("SHAP (SHapley Additive exPlanations) explains the prediction of an instance by showing the contribution of each feature value to the prediction. It calculates Shapley values, which represent the average marginal contribution of a feature value across all possible coalitions of features.")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # the predicted class for the selected instance
    pred_i = int(pred[slider_idx])
    # this illustrates why the model predict this particular outcome
    shap.force_plot(
        explainer.expected_value[pred_i],
        shap_values[pred_i][slider_idx, :],
        X_test.iloc[slider_idx, :],
        matplotlib=True,
    )
    st.pyplot()


def show_local_interpretation(
    X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework
):
    """show the interpretation based on the selected framework"""
    n_data = X_test.shape[0]
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
    """show model performance metrics such as classification report or confusion matrix"""
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
    conf_matrix = confusion_matrix(y_test, pred, labels=list(set(y_test)))
    sns.set(font_scale=1.4)
    sns.heatmap(
        conf_matrix,
        square=True,
        annot=True,
        annot_kws={"size": 15},
        cmap="YlGnBu",
        cbar=False,
    )
    st.sidebar.pyplot()


def draw_pdp(clf, dataset, features, target_labels, dim_model):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""

    if dim_model != "XGBoost":
        selected_col = st.selectbox("Select a feature", features)
        st.info("A Partial Dependence Plot (PDP) shows the marginal effect of one or two features on the predicted outcome of a machine learning model. It helps to visualize how a feature influences the prediction on average.")

        pdp_dist = pdp.pdp_isolate(
            model=clf, dataset=dataset, model_features=features, feature=selected_col
        )
        if len(target_labels) <= 5:
            ncol = len(target_labels)
        else:
            ncol = 5
        pdp.pdp_plot(pdp_dist, selected_col, ncols=ncol, figsize=(12, 5))
        st.pyplot()

def show_global_interpretation_shap(X_train, clf):
    st.info("Global interpretation using SHAP summary plot. This plot shows which features are most important overall and how their values impact the model's output across all predictions.")
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if isinstance(shap_values, list):  # multiclass
        for i, class_shap_values in enumerate(shap_values):
            st.subheader(f"Class {i}")
            shap.summary_plot(class_shap_shap_values, X_train, plot_type="bar", show=False)
            st.pyplot()
    else:  # binary or regression
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot()

def main():
    dim_data = st.sidebar.selectbox(
        "Try out sample data", ("iris", "titanic", "census income")
    )
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")

    df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)

    X_train, X_test, y_train, y_test = splitdata(X, y)
    dim_model = st.sidebar.selectbox(
        "Choose a model", ("XGBoost", "lightGBM", "randomforest")
    )
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "lightGBM":
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(
                class_weight="balanced", objective="multiclass", n_jobs=-1, verbose=-1
            )
        else:
            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "XGBoost":
        params = {
            "max_depth": 5,
            "silent": 1,
            "random_state": 2,
            "num_class": len(target_labels),
        }
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix)
    pred = make_pred(dim_model, X_test, clf)

    dim_framework = st.sidebar.radio(
        "Choose interpretation framework", ["SHAP", "ELI5"]
    )
    if st.sidebar.checkbox("Preview uploaded data"):
        st.sidebar.dataframe(df.head())
    st.sidebar.markdown("#### Classification report")
    show_perf_metrics(y_test, pred)
    st.markdown("#### Global Interpretation")
    st.text("Most important features")
    info_global = st.button("How it is calculated")
    if info_global:
        st.info("Global interpretation aims to understand the overall behavior of the model. It shows which features are generally most important across all predictions.")
    if dim_framework == "SHAP":
        show_global_interpretation_shap(X_train, clf)
    elif dim_framework == "ELI5":
        show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model)

    if st.sidebar.button("About the app"):
        st.sidebar.markdown(
            "This app helps you interpret black-box machine learning models by providing global and local explanations. You can upload your own data or use sample datasets."
        )
        st.sidebar.markdown(
            '<a href="https://ctt.ac/zu8S4"><img src="https://image.flaticon.com/icons/svg/733/733579.svg" width=16></a>',
            unsafe_allow_html=True,
        )
    st.markdown("#### Local Interpretation")

    # misclassified
    if st.checkbox("Filter for misclassified"):
        X_test, y_test, pred = filter_misclassified(X_test, y_test, pred)
        if X_test.shape[0] == 0:
            st.text("No misclassification🎉")
        else:
            st.text(str(X_test.shape[0]) + " misclassified total")
            show_local_interpretation(
                X_test,
                y_test,
                clf,
                pred,
                target_labels,
                features,
                dim_model,
                dim_framework,
            )
    else:
        show_local_interpretation(
            X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework
        )
    if dim_model != "XGBoost" and st.checkbox("Show how features vary with outcome"):
        draw_pdp(clf, X_train, features, target_labels, dim_model)
