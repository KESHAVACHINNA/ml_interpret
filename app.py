import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset
import shap
from io import StringIO

# Title and Subheader
st.title("ML Interpreter")
st.subheader("Blackbox ML classifiers visually explained")

def clean_column_names(columns):
    """Clean column names to be alphanumeric only"""
    return ["".join(c if c.isalnum() else "_" for c in str(x)) for x in columns]

def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            st.error("Error reading file. Please upload a valid CSV file.")
            return None, None, None, None, None
            
        df.columns = clean_column_names(df.columns)
        col_arranged = df.columns.tolist()
        target_col = st.sidebar.selectbox("Choose the target variable", col_arranged)
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
        X.columns = clean_column_names(X.columns)
        features = X.columns
        target_labels = np.array(['<=50K', '>50K'])
        df = pd.concat([X, pd.Series(y, name='Outcome')], axis=1)
        target_col = 'Outcome'
        # Use label encoder for consistency
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    return df, X, y, features, target_labels

def encode_data(data, targetcol):
    """preprocess categorical value"""
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    X.columns = clean_column_names(X.columns)
    features = X.columns
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(data[targetcol])
    target_labels = le.classes_
    
    return X, y, features, target_labels

def splitdata(X, y):
    """split dataset into training & testing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0
    )
    return X_train, X_test, y_train, y_test

def make_pred(dim_model, X_test, clf, target_labels):
    """get y_pred using the classifier"""
    if dim_model == "XGBoost":
        dtest = DMatrix(X_test)
        pred_proba = clf.predict(dtest)
        if len(target_labels) > 2:  # Multiclass
            pred = np.argmax(pred_proba, axis=1)
        else:  # Binary classification
            pred = (pred_proba > 0.5).astype(int)
    elif dim_model == "lightGBM":
        pred = clf.predict(X_test)
    else:  # RandomForest
        pred = clf.predict(X_test)
    return pred

def show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    """show most important features via permutation importance in ELI5"""
    if dim_model == "XGBoost":
        # Use built-in feature importance for XGBoost
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=features.values, top=5
        ).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=5, random_state=1).fit(X_train, y_train)
        df_global_explain = eli5.explain_weights_df(
            perm, feature_names=features.values, top=5
        ).round(2)
    
    if not df_global_explain.empty:
        bar = (
            alt.Chart(df_global_explain)
            .mark_bar(color='steelblue', opacity=0.7)
            .encode(
                x='weight:Q',
                y=alt.Y('feature:N', sort='-x'),
                tooltip=['feature', 'weight']
            )
            .properties(height=300, title='Feature Importances')
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.warning("No feature importance data available")

def show_global_interpretation_shap(clf, X_train, model_type):
    """show most important features via SHAP"""
    try:
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(DMatrix(X_train))
        elif model_type == "lightGBM":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)
        else:  # RandomForest
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_train,
            plot_type="bar",
            max_display=5,
            show=False
        )
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"SHAP error: {str(e)}")

def filter_misclassified(X_test, y_test, pred):
    """get misclassified instances"""
    idx_misclassified = pred != y_test
    return X_test[idx_misclassified], y_test[idx_misclassified], pred[idx_misclassified]

def show_local_interpretation_eli5(
    X_test, clf, target_labels, features, dim_model, instance_idx
):
    """show the interpretation of individual decision points"""
    with st.expander("How this works"):
        st.info("""
        **What's included**  
        - Input data is split 80/20 into training and testing
        - Each testing datapoint can be inspected by index
        **To Read the table**  
        - Contribution: Influence direction (+/-) and magnitude
        - Value: Feature value in the dataset
        - Bias: Model intercept term
        """)
    
    try:
        if dim_model == "XGBoost":
            html_str = eli5.show_prediction(
                clf, 
                X_test.iloc[instance_idx], 
                feature_names=features.tolist(),
                show_feature_values=True,
                top=10
            )
        else:
            html_str = eli5.show_prediction(
                clf,
                X_test.iloc[instance_idx],
                feature_names=features.tolist(),
                target_names=target_labels,
                show_feature_values=True,
                top=10
            )
        st.components.v1.html(html_str.data, height=400)
    except Exception as e:
        st.error(f"ELI5 error: {str(e)}")

def show_local_interpretation_shap(clf, X_test, model_type, instance_idx, target_labels):
    """show the interpretation of individual decision points"""
    with st.expander("How this works"):
        st.info("""
        **Force Plot Explanation**  
        - Red features: Push prediction higher  
        - Blue features: Push prediction lower  
        - Base value: Average model output  
        - Output value: Model prediction for this instance  
        **Note**: For multiclass, this shows the prediction for the selected class
        """)
    
    try:
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(DMatrix(X_test))
            instance = X_test.iloc[instance_idx:instance_idx+1]
            
            if len(target_labels) > 2:
                # For multiclass, show prediction for the actual class
                class_idx = int(y_test[instance_idx])
                shap.force_plot(
                    explainer.expected_value[class_idx],
                    shap_values[class_idx][instance_idx],
                    instance,
                    matplotlib=True,
                    show=False,
                    figsize=(15, 3)
                )
            else:
                shap_values = explainer.shap_values(DMatrix(instance))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    instance,
                    matplotlib=True,
                    show=False,
                    figsize=(15, 3)
                )
                
        elif model_type == "lightGBM" or model_type == "randomforest":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            instance = X_test.iloc[instance_idx:instance_idx+1]
            
            if len(target_labels) > 2:
                class_idx = int(y_test[instance_idx])
                shap.force_plot(
                    explainer.expected_value[class_idx],
                    shap_values[class_idx][instance_idx],
                    instance,
                    matplotlib=True,
                    show=False,
                    figsize=(15, 3)
                )
            else:
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[instance_idx],
                    instance,
                    matplotlib=True,
                    show=False,
                    figsize=(15, 3)
                )
                
        st.pyplot(bbox_inches='tight')
        plt.clf()
    except Exception as e:
        st.error(f"SHAP error: {str(e)}")

def show_perf_metrics(y_test, pred, target_labels):
    """show model performance metrics"""
    st.sidebar.subheader("Classification Report")
    report = classification_report(y_test, pred, target_names=target_labels, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))
    
    st.sidebar.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_labels,
        yticklabels=target_labels,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.sidebar.pyplot(fig)
    plt.clf()

def draw_pdp(clf, X_train, features, target_labels, dim_model):
    """draw PDP plot"""
    if dim_model == "XGBoost":
        st.warning("PDP plots not currently supported for XGBoost in this app")
        return
        
    selected_col = st.selectbox("Select a feature for PDP", features)
    
    with st.expander("About Partial Dependence Plots"):
        st.info("""
        **Interpretation Guide**  
        - Shows how a feature affects model predictions  
        - Y-axis: Change in predicted probability  
        - X-axis: Feature values  
        **Key Insights**  
        - Upward slope: Higher values increase prediction probability  
        - Downward slope: Lower values increase prediction probability  
        - Flat line: Little predictive relationship  
        """)
    
    try:
        pdp_iso = pdp.pdp_isolate(
            model=clf, 
            dataset=X_train, 
            model_features=features, 
            feature=selected_col
        )
        
        fig, ax = pdp.pdp_plot(
            pdp_iso, 
            selected_col, 
            center=True,
            plot_lines=True,
            frac_to_plot=100,
            x_quantile=True,
            show_percentile=True,
            figsize=(10, 6)
        )
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"PDP Error: {str(e)}")

def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    ################################################
    # upload file
    ################################################
    st.sidebar.header("Data Configuration")
    dim_data = st.sidebar.selectbox(
        "Try sample data", 
        ("iris", "titanic", "census income"),
        index=0
    )
    uploaded_file = st.sidebar.file_uploader("Or upload CSV", type="csv")
    
    if uploaded_file or dim_data:
        df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)
        
        if df is None:
            st.warning("Please upload data or select sample dataset")
            return
            
        st.session_state.data_loaded = True
        st.sidebar.success(f"Data loaded! {X.shape[0]} samples with {X.shape[1]} features")
        
        if st.sidebar.checkbox("Preview data"):
            st.sidebar.dataframe(df.head(3))
    
    if not st.session_state.data_loaded:
        st.info("Please select sample data or upload a CSV file to begin")
        return
        
    ################################################
    # process data
    ################################################
    X_train, X_test, y_train, y_test = splitdata(X, y)
    
    ################################################
    # apply model
    ################################################
    st.sidebar.header("Model Configuration")
    dim_model = st.sidebar.selectbox(
        "Choose model", 
        ("randomforest", "lightGBM", "XGBoost"),
        index=0
    )
    
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "lightGBM":
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, verbose=-1)
        else:
            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "XGBoost":
        params = {
            "objective": "multi:softprob" if len(target_labels) > 2 else "binary:logistic",
            "max_depth": 4,
            "silent": 1,
            "random_state": 2,
            "num_class": len(target_labels) if len(target_labels) > 2 else 1
        }
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix, num_boost_round=100)
    
    ################################################
    # Predict
    ################################################
    pred = make_pred(dim_model, X_test, clf, target_labels)
    
    st.sidebar.header("Interpretation Framework")
    dim_framework = st.sidebar.radio(
        "Choose explanation method", 
        ["SHAP", "ELI5"],
        index=0
    )
    
    ################################################
    # Model output
    ################################################
    show_perf_metrics(y_test, pred, target_labels)
    
    ################################################
    # Global Interpretation
    ################################################
    st.header("Global Interpretation")
    st.subheader("Feature Importance Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        **What this shows**:  
        - Most influential features in the model  
        - Direction of influence (positive/negative)  
        - Relative importance magnitude  
        """)
    
    with col2:
        if dim_framework == "SHAP":
            show_global_interpretation_shap(clf, X_train, dim_model)
        elif dim_framework == "ELI5":
            show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model)
    
    ################################################
    # Local Interpretation
    ################################################
    st.header("Local Interpretation")
    st.subheader("Individual Prediction Explanations")
    
    # Misclassification filter
    misclassified_only = st.checkbox("Show misclassified only", value=False)
    if misclassified_only:
        X_test_disp, y_test_disp, pred_disp = filter_misclassified(X_test, y_test, pred)
        if X_test_disp.shape[0] == 0:
            st.success("🎉 No misclassifications found!")
            return
    else:
        X_test_disp, y_test_disp, pred_disp = X_test, y_test, pred
    
    n_data = X_test_disp.shape[0]
    instance_idx = st.slider("Select datapoint to explain", 0, n_data - 1, 0, key='instance_slider')
    
    actual = target_labels[y_test_disp.iloc[instance_idx]]
    predicted = target_labels[pred_disp[instance_idx]]
    
    st.markdown(f"""
    **Prediction Details**  
    - Actual: `{actual}`  
    - Predicted: `{predicted}`  
    - Correct: {"✅" if actual == predicted else "❌"}  
    """)
    
    if dim_framework == "SHAP":
        show_local_interpretation_shap(clf, X_test_disp, dim_model, instance_idx, target_labels)
    elif dim_framework == "ELI5":
        show_local_interpretation_eli5(
            X_test_disp, clf, target_labels, features, dim_model, instance_idx
        )
    
    ################################################
    # PDP plot
    ################################################
    if st.checkbox("Show Partial Dependence Plots", value=False):
        st.header("Partial Dependence Analysis")
        draw_pdp(clf, X_train, features, target_labels, dim_model)
    
    ################################################
    # Footer
    ################################################
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    

if __name__ == "__main__":
    main()
