
#  ML Interpretability App

A Streamlit web app to visualize and understand how machine learning models make predictions using SHAP and ELI5.

🚀 **Live Demo**: [mlinterpret.streamlit.app](https://mlinterpret-jdjam9qwhvsd84htvtbjbu.streamlit.app/)

---

## 💡 Features

- 🧪 Select and train classifiers (Random Forest, XGBoost, LightGBM) on the Iris dataset.
- 📊 View classification reports and sample data.
- 🔍 Visualize global feature importance using **SHAP beeswarm plots**.
- 🔬 Understand local predictions using **SHAP force plots**.
- 🧠 Analyze feature weights with **ELI5** permutation importance.

---

## 📦 Requirements

```bash
pip install streamlit shap eli5 scikit-learn xgboost lightgbm matplotlib seaborn altair joblib
