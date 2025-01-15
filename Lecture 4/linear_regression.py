import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/nileshely/SuperStore-Dataset-2019-2022/refs/heads/main/superstore_dataset.csv"
data = pd.read_csv(url)

# Clean column names
data.columns = data.columns.str.strip()

# Data preprocessing: Select relevant columns
# Objective: Relationship between 'sales' (target variable) and 'quantity' (feature)
X = data[['quantity']].values  # Independent variable
y = data['sales'].values  # Target variable

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual Values", alpha=0.5)
plt.plot(X_test, y_pred, color="red", label="Predictions")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.title("Linear Regression: Sales vs. Quantity")
plt.legend()
plt.show()
