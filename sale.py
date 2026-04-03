import streamlit as st
import joblib
import numpy as np

# Load saved models and scaler
import joblib

linear_model = joblib.load("C:/Users/amand/Downloads/linear_model.pkl")
ridge_model = joblib.load("C:/Users/amand/Downloads/ridge_model.pkl")
lasso_model = joblib.load("C:/Users/amand/Downloads/lasso_model.pkl")
scaler = joblib.load("C:/Users/amand/Downloads/scaler (1).pkl")

print("All Models Loaded Successfully")

# Page config
st.set_page_config(page_title="Sales Prediction App", page_icon="📊", layout="centered")

# Title
st.title("📊 Advertising Sales Prediction App")
st.write("Predict Sales based on TV, Radio, and Newspaper advertising budgets.")

st.markdown("---")

# Sidebar
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox(
    "Select Regression Model",
    ["Linear Regression", "Ridge Regression", "Lasso Regression"]
)

st.sidebar.markdown("---")
st.sidebar.info("Enter advertising budgets to predict sales.")

# Input fields
st.subheader("Enter Advertising Budget")

tv = st.number_input("TV Ad Budget ($)", min_value=0.0, value=100.0, step=1.0)
radio = st.number_input("Radio Ad Budget ($)", min_value=0.0, value=25.0, step=1.0)
newspaper = st.number_input("Newspaper Ad Budget ($)", min_value=0.0, value=20.0, step=1.0)

# Prediction
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_scaled)[0]
    elif model_choice == "Ridge Regression":
        prediction = ridge_model.predict(input_scaled)[0]
    else:
        prediction = lasso_model.predict(input_scaled)[0]

    st.success(f"📈 Predicted Sales using {model_choice}: {prediction:.2f}")

    # Compare all models
    st.subheader("Compare All Models")
    pred_lr = linear_model.predict(input_scaled)[0]
    pred_ridge = ridge_model.predict(input_scaled)[0]
    pred_lasso = lasso_model.predict(input_scaled)[0]

    st.write(f"**Linear Regression:** {pred_lr:.2f}")
    st.write(f"**Ridge Regression:** {pred_ridge:.2f}")
    st.write(f"**Lasso Regression:** {pred_lasso:.2f}")

st.markdown("---")
st.caption("Built with Streamlit | Sales Prediction Project")