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
st.set_page_config(page_title="ML Interpreter", layout="wide")
st.title("ML Interpreter")
st.subheader("Blackbox ML classifiers visually explained")

def clean_column_names(columns):
    """Clean column names to be alphanumeric only"""
    return ["".join(c if c.isalnum() else "_" for c in str(x)) for x in columns]

def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = clean_column_names(df.columns)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None, None, None, None, None
            
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
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    return df, X, y, features, target_labels

def encode_data(data, targetcol):
    """preprocess categorical value"""
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    X.columns = clean_column_names(X.columns)
    features = X.columns
    
    le = LabelEncoder()
    y = le.fit_transform(data[targetcol])
    target_labels = le.classes_
    
    return X, y, features, target_labels

def splitdata(X, y):
    """split dataset into training & testing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=42, stratify=y
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
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=features.tolist(), top=10
        ).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=5, random_state=1).fit(X_train, y_train)
        df_global_explain = eli5.explain_weights_df(
            perm, feature_names=features.tolist(), top=10
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
            .properties(height=400, title='Top 10 Feature Importances')
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.warning("No feature importance data available")

def show_global_interpretation_shap(clf, X_train, model_type):
    """show most important features via SHAP"""
    try:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(DMatrix(X_train))
        else:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_train,
            plot_type="bar",
            max_display=10,
            show=False
        )
        st.pyplot(bbox_inches='tight')
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
    with st.expander("ℹ️ How to interpret this"):
        st.info("""
        **Understanding the Explanation**  
        - **Contribution**: How much each feature affected the prediction  
        - **Value**: The actual feature value in this instance  
        - **Bias**: The model's baseline prediction  
        
        **Colors**:  
        - Green: Features pushing prediction higher  
        - Red: Features pushing prediction lower  
        """)
    
    try:
        if dim_model == "XGBoost":
            html_str = eli5.show_prediction(
                clf, 
                X_test.iloc[instance_idx], 
                feature_names=features.tolist(),
                show_feature_values=True,
                top=15
            )
        else:
            html_str = eli5.show_prediction(
                clf,
                X_test.iloc[instance_idx],
                feature_names=features.tolist(),
                target_names=target_labels.tolist(),
                show_feature_values=True,
                top=15
            )
        st.components.v1.html(html_str.data, height=500, scrolling=True)
    except Exception as e:
        st.error(f"ELI5 error: {str(e)}")

def show_local_interpretation_shap(clf, X_test, model_type, instance_idx, target_labels):
    """show the interpretation of individual decision points"""
    with st.expander("ℹ️ How to interpret this force plot"):
        st.info("""
        **Force Plot Guide**  
        - Base value: Model's average prediction  
        - Output value: Prediction for this instance  
        - Red arrows: Increase prediction  
        - Blue arrows: Decrease prediction  
        - Length: Strength of feature's effect  
        """)
    
    try:
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(DMatrix(X_test))
            instance = X_test.iloc[instance_idx:instance_idx+1]
            
            if len(target_labels) > 2:
                pred_class = clf.predict(DMatrix(instance))[0]
                class_idx = int(np.argmax(pred_class))
                shap.initjs()
                plt.figure()
                shap.force_plot(
                    explainer.expected_value[class_idx],
                    shap_values[class_idx][instance_idx],
                    instance,
                    feature_names=X_test.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
            else:
                shap.initjs()
                plt.figure()
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[instance_idx],
                    instance,
                    feature_names=X_test.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
                
        else:  # lightGBM or RandomForest
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            instance = X_test.iloc[instance_idx:instance_idx+1]
            
            if len(target_labels) > 2:
                pred_class = clf.predict(instance)[0]
                class_idx = int(pred_class)
                shap.initjs()
                plt.figure()
                shap.force_plot(
                    explainer.expected_value[class_idx],
                    shap_values[class_idx][instance_idx],
                    instance,
                    feature_names=X_test.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
            else:
                shap.initjs()
                plt.figure()
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[instance_idx],
                    instance,
                    feature_names=X_test.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
        
        st.pyplot(bbox_inches='tight')
        plt.clf()
    except Exception as e:
        st.error(f"SHAP error: {str(e)}")
        st.error("This might occur with certain model types or data formats.")

def show_perf_metrics(y_test, pred, target_labels):
    """show model performance metrics"""
    st.sidebar.subheader("📊 Classification Report")
    report = classification_report(y_test, pred, target_names=target_labels, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))
    
    st.sidebar.subheader("📈 Confusion Matrix")
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
    ax.set_title('Confusion Matrix')
    st.sidebar.pyplot(fig)
    plt.clf()

def draw_pdp(clf, X_train, features, target_labels, dim_model):
    """draw PDP plot"""
    if dim_model == "XGBoost":
        st.warning("PDP plots not currently supported for XGBoost in this app")
        return
        
    selected_col = st.selectbox("Select a feature for analysis", features)
    
    with st.expander("ℹ️ About Partial Dependence Plots"):
        st.info("""
        **What This Shows**  
        - How a feature affects predictions across its value range  
        - Marginal relationship between feature and outcome  
        
        **Interpretation**  
        - Upward slope: Higher values increase prediction probability  
        - Downward slope: Lower values increase probability  
        - Flat line: Little predictive relationship  
        """)
    
    try:
        pdp_iso = pdp.pdp_isolate(
            model=clf, 
            dataset=X_train, 
            model_features=features, 
            feature=selected_col
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pdp.pdp_plot(
            pdp_iso, 
            selected_col, 
            center=True,
            plot_lines=True,
            frac_to_plot=100,
            x_quantile=True,
            show_percentile=True,
            ax=ax
        )
        ax.set_title(f'Partial Dependence Plot for {selected_col}')
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"PDP Error: {str(e)}")

def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    ################################################
    # Sidebar Configuration
    ################################################
    st.sidebar.header("⚙️ Configuration")
    
    # Data selection
    dim_data = st.sidebar.selectbox(
        "Choose sample data", 
        ("iris", "titanic", "census income"),
        index=0
    )
    uploaded_file = st.sidebar.file_uploader("Or upload your CSV", type="csv")
    
    # Model selection
    dim_model = st.sidebar.selectbox(
        "Select model", 
        ("randomforest", "lightGBM", "XGBoost"),
        index=0
    )
    
    # Interpretation framework
    dim_framework = st.sidebar.radio(
        "Interpretation method", 
        ["SHAP", "ELI5"],
        index=0
    )
    
    ################################################
    # Data Loading and Processing
    ################################################
    if uploaded_file or dim_data:
        with st.spinner("Loading and processing data..."):
            df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)
            
            if df is not None:
                st.session_state.data_loaded = True
                X_train, X_test, y_train, y_test = splitdata(X, y)
                
                # Model training
                with st.spinner(f"Training {dim_model} model..."):
                    if dim_model == "randomforest":
                        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        clf.fit(X_train, y_train)
                    elif dim_model == "lightGBM":
                        if len(target_labels) > 2:
                            clf = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, random_state=42)
                        else:
                            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, random_state=42)
                        clf.fit(X_train, y_train)
                    elif dim_model == "XGBoost":
                        params = {
                            "objective": "multi:softprob" if len(target_labels) > 2 else "binary:logistic",
                            "max_depth": 4,
                            "random_state": 42,
                            "num_class": len(target_labels) if len(target_labels) > 2 else 1
                        }
                        dmatrix = DMatrix(data=X_train, label=y_train)
                        clf = xgb.train(params=params, dtrain=dmatrix, num_boost_round=100)
                
                # Make predictions
                pred = make_pred(dim_model, X_test, clf, target_labels)
                
                # Show performance metrics
                show_perf_metrics(y_test, pred, target_labels)
    
    if not st.session_state.get('data_loaded', False):
        st.info("👈 Please select sample data or upload a CSV file to begin")
        return
    
    ################################################
    # Main Content
    ################################################
    tab1, tab2, tab3 = st.tabs(["Global Analysis", "Local Analysis", "Feature Analysis"])
    
    with tab1:
        st.header("🌍 Global Model Interpretation")
        st.write("Understand what features drive your model's predictions overall")
        
        if dim_framework == "SHAP":
            show_global_interpretation_shap(clf, X_train, dim_model)
        else:
            show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model)
    
    with tab2:
        st.header("🔍 Local Prediction Explanation")
        st.write("Explore how individual predictions are made")
        
        # Misclassification filter
        misclassified_only = st.checkbox("Show only misclassified instances", value=False)
        if misclassified_only:
            X_test_disp, y_test_disp, pred_disp = filter_misclassified(X_test, y_test, pred)
            if X_test_disp.shape[0] == 0:
                st.success("🎉 No misclassifications found!")
                st.stop()
        else:
            X_test_disp, y_test_disp, pred_disp = X_test, y_test, pred
        
        n_data = X_test_disp.shape[0]
        instance_idx = st.slider(
            "Select instance to explain", 
            0, 
            n_data - 1, 
            0,
            key='instance_slider'
        )
        
        # Prediction details
        actual = target_labels[y_test_disp.iloc[instance_idx]]
        predicted = target_labels[pred_disp[instance_idx]]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual", actual)
        with col2:
            st.metric("Predicted", predicted, delta="Correct" if actual == predicted else "Incorrect", 
                     delta_color="normal" if actual == predicted else "inverse")
        
        if dim_framework == "SHAP":
            show_local_interpretation_shap(clf, X_test_disp, dim_model, instance_idx, target_labels)
        else:
            show_local_interpretation_eli5(
                X_test_disp, clf, target_labels, features, dim_model, instance_idx
            )
    
    with tab3:
        st.header("📊 Feature Behavior Analysis")
        st.write("Understand how individual features affect predictions")
        draw_pdp(clf, X_train, features, target_labels, dim_model)
    
    ################################################
    # Footer
    ################################################
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About this app**  
    [GitHub Repository](https://github.com/yourusername/ml-interpreter)  
    Built with Streamlit · SHAP · ELI5  
    For educational and demonstration purposes  
    """)

if __name__ == "__main__":
    main()
