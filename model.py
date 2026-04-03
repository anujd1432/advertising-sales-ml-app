import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df  = pd.read_csv("C:/Users/amand/Downloads/advertising.csv")

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
lin = LinearRegression()
ridge = Ridge()
lasso = Lasso()

lin.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Save models
joblib.dump(lin, "linear.pkl")
joblib.dump(ridge, "ridge.pkl")
joblib.dump(lasso, "lasso.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models Saved Successfully")