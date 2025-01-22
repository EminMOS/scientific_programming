import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder

url = "https://raw.githubusercontent.com/nileshely/SuperStore-Dataset-2019-2022/refs/heads/main/superstore_dataset.csv"
data = pd.read_csv(url)

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

print("Columns in the dataset:", data.columns)

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

data.fillna(0, inplace=True)

categorical_columns = ['category', 'subcategory', 'region', 'segment']
label_encoders = {}
for col in categorical_columns:
    if col in data.columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    else:
        print(f"Column '{col}' not found in the dataset.")

# Task 1: 
features_sales = ['category', 'subcategory', 'region', 'quantity', 'segment']
target_sales = 'sales'

X_sales = data[features_sales]
y_sales = data[target_sales]

X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42
)

sales_model = RandomForestRegressor(random_state=42)
sales_model.fit(X_train_sales, y_train_sales)

sales_predictions = sales_model.predict(X_test_sales)
print("\nSales Prediction RMSE:", np.sqrt(mean_squared_error(y_test_sales, sales_predictions)))

# Task 2: 
features_profit = ['category', 'subcategory', 'region', 'quantity', 'segment']
target_profit = 'profit'

X_profit = data[features_profit]
y_profit = data[target_profit]

X_train_profit, X_test_profit, y_train_profit, y_test_profit = train_test_split(
    X_profit, y_profit, test_size=0.2, random_state=42
)

profit_model = RandomForestRegressor(random_state=42)
profit_model.fit(X_train_profit, y_train_profit)

profit_predictions = profit_model.predict(X_test_profit)
print("\nProfit Prediction RMSE:", np.sqrt(mean_squared_error(y_test_profit, profit_predictions)))

# Task 3:
profit_threshold = y_profit.median()
data['profit_category'] = (data['profit'] > profit_threshold).astype(int)

features_profit_class = ['category', 'subcategory', 'region', 'quantity', 'segment']
target_profit_class = 'profit_category'

X_profit_class = data[features_profit_class]
y_profit_class = data[target_profit_class]

X_train_profit_class, X_test_profit_class, y_train_profit_class, y_test_profit_class = train_test_split(
    X_profit_class, y_profit_class, test_size=0.2, random_state=42
)

profit_class_model = RandomForestClassifier(random_state=42)
profit_class_model.fit(X_train_profit_class, y_train_profit_class)

profit_class_predictions = profit_class_model.predict(X_test_profit_class)
print("\nProfit Classification Report:")
print(classification_report(y_test_profit_class, profit_class_predictions, zero_division=0))

# Task 4: 
sales_threshold = y_sales.median()
data['sales_category'] = (data['sales'] > sales_threshold).astype(int)

features_sales_class = ['category', 'subcategory', 'region', 'quantity', 'segment']
target_sales_class = 'sales_category'

X_sales_class = data[features_sales_class]
y_sales_class = data[target_sales_class]

X_train_sales_class, X_test_sales_class, y_train_sales_class, y_test_sales_class = train_test_split(
    X_sales_class, y_sales_class, test_size=0.2, random_state=42
)

sales_class_model = RandomForestClassifier(random_state=42)
sales_class_model.fit(X_train_sales_class, y_train_sales_class)

sales_class_predictions = sales_class_model.predict(X_test_sales_class)
print("\nSales Classification Report:")
print(classification_report(y_test_sales_class, sales_class_predictions, zero_division=0))

sales_importance = sales_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(features_sales, sales_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance for Sales Prediction")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

profit_importance = profit_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(features_profit, profit_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance for Profit Prediction")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()