import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load saved models, scaler, and test data
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
gb_model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# App title and description
st.title("White Wine Quality Classification")
st.write(
    """
    This app predicts the quality of white wine based on its chemical properties. 
    You can select a model, input feature values via sliders, and view the results along with detailed model insights.
    """
)

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Select a Model", 
    ("Random Forest", "SVM", "Gradient Boosting")
)

# Sidebar for input features
st.sidebar.header("Input Features")

# Define feature ranges and default values
feature_ranges = {
    "fixed acidity": (3.0, 15.0, 7.0),            # (min, max, default)
    "volatile acidity": (0.1, 1.5, 0.27),
    "citric acid": (0.0, 1.0, 0.36),
    "residual sugar": (0.1, 65.0, 20.7),
    "chlorides": (0.01, 0.1, 0.045),
    "free sulfur dioxide": (1.0, 100.0, 45.0),
    "total sulfur dioxide": (10.0, 300.0, 170.0),
    "density": (0.9900, 1.0100, 1.0010),
    "pH": (2.5, 4.5, 3.00),
    "sulphates": (0.1, 1.5, 0.45),
    "alcohol": (8.0, 15.0, 8.8),
}

# Dynamically generate sliders for all features
feature_inputs = {}
for feature, (min_val, max_val, default_val) in feature_ranges.items():
    feature_inputs[feature] = st.sidebar.slider(
        label=f"{feature}:",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
    )

# Convert inputs to DataFrame
input_data = pd.DataFrame([feature_inputs])

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict quality based on selected model
if model_option == "Random Forest":
    prediction = rf_model.predict(input_scaled)
    model = rf_model
elif model_option == "SVM":
    prediction = svm_model.predict(input_scaled)
    model = svm_model
elif model_option == "Gradient Boosting":
    prediction = gb_model.predict(input_scaled)
    model = gb_model

# Display prediction result
st.subheader("Prediction Result")
st.write(f"**Predicted Quality:** {int(prediction[0])}")
st.write(f"**Selected Model:** {model_option}")

# Evaluate models on training/testing data
st.subheader("Model Insights")

# Display Classification Report
st.write("### Classification Report")
st.text(classification_report(y_test, model.predict(X_test)))

# Display Confusion Matrix
st.write("### Confusion Matrix")
conf_matrix = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
st.pyplot(conf_matrix.figure_)

# Feature Importance Visualization (For RF and GB models)
if model_option in ["Random Forest", "Gradient Boosting"]:
    st.write("### Feature Importance")
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(list(feature_ranges.keys()), feature_importances, color="skyblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for {model_option}")
    st.pyplot(plt)

# Add histogram visualizations for feature distributions
st.write("### Feature Distributions in Dataset")
fig, ax = plt.subplots(len(feature_ranges), 1, figsize=(8, len(feature_ranges) * 2))
for i, col in enumerate(feature_ranges.keys()):
    ax[i].hist(X_test[:, i], bins=20, color="lightgreen", edgecolor="black")
    ax[i].set_title(col)
    ax[i].set_xlabel("Value")
    ax[i].set_ylabel("Frequency")
st.pyplot(fig)

# Footer
st.sidebar.write("---")
st.sidebar.write(
    "Developed by: Jeffery Kuo | Powered by Streamlit"
)
