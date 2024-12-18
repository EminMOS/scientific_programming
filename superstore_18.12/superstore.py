import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/nileshely/SuperStore-Dataset-2019-2022/refs/heads/main/superstore_dataset.csv"

data = pd.read_csv(url)
print("\nAll data: ")
print(data)
print ("\nFirst 10 rows: ")
print(data.head(10))

print("\nData description: ")
print(data.describe())

print("\nMissing values before cleaning:")
print(data.isnull().sum())

#missing values
data.fillna(
    {col: 0 if data[col].dtype in ["float64", "int64"] else "Unknown" for col in data.columns},
    inplace=True
)

# Display cleaned dataset's missing values
print("\nMissing values after cleaning:")
print(data.isnull().sum())


if "order_date" in data.columns:
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
if "ship_date" in data.columns:
    data["ship_date"] = pd.to_datetime(data["ship_date"], errors="coerce")

# Create a new column called Revenue if it’s missing
if "Revenue" not in data.columns and "quantity" in data.columns and "sales" in data.columns:
    data["Revenue"] = data["quantity"].astype(float) * data["sales"].astype(float)

# Display Revenue column and basic statistics if created
if "Revenue" in data.columns:
    print("\nRevenue column created successfully.")
    print(data[["Revenue"]].describe())

# Find the customer who placed the most orders
if "customer" in data.columns:
    top_customer = data["customer"].value_counts().idxmax()
    top_customer_orders = data["customer"].value_counts().max()
    print(f"\nCustomer who placed the most orders: {top_customer} with {top_customer_orders} orders")

# Data Aggregation
if "product_name" in data.columns and "quantity" in data.columns:
    product_quantity = data.groupby("product_name")["quantity"].sum().sort_values(ascending=False)
    print("\nTotal Quantity Sold by Product:")
    print(product_quantity)

# show daily revenue trends
if "order_date" in data.columns and "Revenue" in data.columns:
    daily_revenue = data.groupby("order_date")["Revenue"].sum()
    print("\nDaily Revenue Trends:")
    print(daily_revenue)

# Visualization top 5
if "product_name" in data.columns and "Revenue" in data.columns:
    top_5_products_by_revenue = data.groupby("product_name")["Revenue"].sum().sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 6))
    top_5_products_by_revenue.plot(kind='bar', color='skyblue')
    plt.title("Top 5 Products by Revenue")
    plt.ylabel("Revenue")
    plt.xlabel("Product Name")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# line graph to visualize daily revenue trends
if "order_date" in data.columns and "Revenue" in data.columns:
    plt.figure(figsize=(12, 6))
    daily_revenue.plot(kind='line', color='green')
    plt.title("Daily Revenue Trends")
    plt.ylabel("Revenue")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# calculate a 7-day moving average for daily revenue
if "order_date" in data.columns and "Revenue" in data.columns:
    daily_revenue_df = daily_revenue.reset_index()
    daily_revenue_df["7-Day Moving Average"] = daily_revenue_df["Revenue"].rolling(window=7).mean()
    print("\nDaily Revenue with 7-Day Moving Average:")
    print(daily_revenue_df)

    # Plot daily revenue with 7-day moving average
    plt.figure(figsize=(12, 6))
    plt.plot(daily_revenue_df["order_date"], daily_revenue_df["Revenue"], label="Daily Revenue", color="blue")
    plt.plot(daily_revenue_df["order_date"], daily_revenue_df["7-Day Moving Average"], label="7-Day Moving Average", color="orange")
    plt.title("Daily Revenue with 7-Day Moving Average")
    plt.ylabel("Revenue")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Identify the day with the highest revenue and the product
if "Revenue" in data.columns and "order_date" in data.columns:
    max_revenue_day = daily_revenue.idxmax()
    max_revenue = daily_revenue.max()
    print(f"\nDay with Highest Revenue: {max_revenue_day}, Revenue: {max_revenue}")

    product_contribution = data[data["order_date"] == max_revenue_day]
    if "product_name" in product_contribution.columns:
        top_product = product_contribution.groupby("product_name")["Revenue"].sum().sort_values(ascending=False).idxmax()
        print(f"Product contributing the most on {max_revenue_day}: {top_product}")
