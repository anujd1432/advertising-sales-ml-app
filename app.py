import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advertising Sales Prediction")

# Load models
lin = joblib.load("linear.pkl")
ridge = joblib.load("ridge.pkl")
lasso = joblib.load("lasso.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Advertising Sales Prediction App")

st.sidebar.header("Enter Advertising Budget")

tv = st.sidebar.slider("TV Budget", 0, 300, 100)
radio = st.sidebar.slider("Radio Budget", 0, 50, 25)
news = st.sidebar.slider("Newspaper Budget", 0, 100, 20)

data = np.array([[tv, radio, news]])
data_scaled = scaler.transform(data)

lin_pred = lin.predict(data_scaled)[0]
ridge_pred = ridge.predict(data_scaled)[0]
lasso_pred = lasso.predict(data_scaled)[0]

st.subheader("Predicted Sales")

st.success(f"Linear Regression: {lin_pred:.2f}")
st.success(f"Ridge Regression: {ridge_pred:.2f}")
st.success(f"Lasso Regression: {lasso_pred:.2f}")

# Graph
models = ["Linear", "Ridge", "Lasso"]
values = [lin_pred, ridge_pred, lasso_pred]

fig, ax = plt.subplots()
ax.bar(models, values)
ax.set_ylabel("Predicted Sales")
ax.set_title("Model Comparison")

st.pyplot(fig)