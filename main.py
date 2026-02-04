# ================================
# House Price Prediction Project
# Using Linear Regression
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
# Change filename as per your Kaggle dataset
data = pd.read_csv("house_price.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# 3. Data Preprocessing

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Convert categorical columns (like location) to numerical
data = pd.get_dummies(data, drop_first=True)

# 4. Define Features and Target
# Change 'price' to the actual target column name if needed
X = data.drop("price", axis=1)
y = data["price"]

# 5. Split Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make Predictions
y_pred = model.predict(X_test)

# 8. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# 9. Visualize Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 10. Predict Price for New Data (Example)
# Make sure the order & number of reminder features matches X columns
sample_house = X.iloc[0:1]
predicted_price = model.predict(sample_house)

print("\nPredicted Price for Sample House:", predicted_price[0])
