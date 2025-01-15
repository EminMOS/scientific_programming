# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns

# Load the dataset directly from seaborn (it's available online)
df = sns.load_dataset('tips')

# Inspect the dataset
print(df.head())

# Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Select target variable (Total Bill) and features
X = df_encoded.drop(columns=['tip'])
y = df_encoded['tip']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict using Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)  # MSE, no squared argument needed
#rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)  # RMSE, use squared=False
print(f'Linear Regression MAE: {mae_lr}')
print(f'Linear Regression MSE: {mse_lr}')
#print(f'Linear Regression RMSE: {rmse_lr}')




# Model 2: Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict using Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)  # MSE, no squared argument needed
#rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)  # RMSE, use squared=False
print(f'Decision Tree MAE: {mae_dt}')
print(f'Decision Tree MSE: {mse_dt}')
#print(f'Decision Tree RMSE: {rmse_dt}')

from sklearn.ensemble import RandomForestRegressor
# Model 3: Random Forest Regression (Non-linear Model)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict using Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
#rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest MSE: {mse_rf}')
#print(f'Random Forest RMSE: {rmse_rf}')

# Visualization: Comparison of Actual vs Predicted Tip (for Decision Tree, Linear Regression, and Random Forest)
plt.figure(figsize=(18,6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Linear Regression: Actual vs Predicted Tip')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')

# Decision Tree
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Decision Tree: Actual vs Predicted Tip')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')

# Random Forest
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Random Forest: Actual vs Predicted Tip')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')

plt.tight_layout()
plt.show()

# Feature Importance for Random Forest
feature_importance = rf_model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importance (Random Forest)')
plt.show()

# Residuals for Random Forest (for example)
residuals_rf = y_test - y_pred_rf
plt.scatter(y_pred_rf, residuals_rf)
plt.hlines(y=0, xmin=y_pred_rf.min(), xmax=y_pred_rf.max(), color='red', lw=2)
plt.title('Residuals for Random Forest')
plt.xlabel('Predicted Tip')
plt.ylabel('Residuals')
plt.show()
