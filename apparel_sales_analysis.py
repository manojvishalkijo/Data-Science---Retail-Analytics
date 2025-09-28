
# Apparel Shop Sales Analysis

## 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('apparel_shop_sales.csv')
df.head()

## 2. Basic Info
df.info()
df.describe()

## 3. Sales Analysis
# Total sales by product category
sales_by_category = df.groupby('ProductCategory')['TotalAmount'].sum().sort_values(ascending=False)
sales_by_category.plot(kind='bar', figsize=(8,5), title='Total Sales by Category')

# Daily sales trend
daily_sales = df.groupby('Date')['TotalAmount'].sum()
daily_sales.plot(kind='line', figsize=(10,5), title='Daily Sales Trend')

## 4. Customer Analysis
# Gender-wise purchases
sns.countplot(x='Gender', data=df)
plt.title('Purchase Count by Gender')
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Customer Age Distribution')
plt.show()

## 5. Inventory Insights
# Popular sizes
sns.countplot(x='Size', data=df)
plt.title('Most Popular Sizes')
plt.show()

# Popular colors
sns.countplot(x='Color', data=df, order=df['Color'].value_counts().index)
plt.title('Most Popular Colors')
plt.show()

## 6. Store Performance
store_sales = df.groupby('StoreLocation')['TotalAmount'].sum().sort_values(ascending=False)
store_sales.plot(kind='bar', figsize=(8,5), title='Sales by Store Location')

## 7. Payment Analysis
sns.countplot(x='PaymentMode', data=df, order=df['PaymentMode'].value_counts().index)
plt.title('Payment Mode Preferences')
plt.show()

## 8. Predictive Analysis (Example: Simple Regression for Sales Forecasting)
from sklearn.linear_model import LinearRegression
import numpy as np

# Convert Date to ordinal
df['Date'] = pd.to_datetime(df['Date'])
df['DayNumber'] = df['Date'].map(pd.Timestamp.toordinal)

# Group by DayNumber
daily = df.groupby('DayNumber')['TotalAmount'].sum().reset_index()

X = daily[['DayNumber']]
y = daily['TotalAmount']

model = LinearRegression()
model.fit(X, y)

# Predict next 7 days
future_days = np.array(range(daily['DayNumber'].max()+1, daily['DayNumber'].max()+8)).reshape(-1,1)
predictions = model.predict(future_days)

plt.figure(figsize=(10,5))
plt.plot(daily['DayNumber'], y, label='Actual Sales')
plt.plot(future_days, predictions, label='Predicted Sales', linestyle='--')
plt.legend()
plt.title('Sales Forecasting (Linear Regression)')
plt.show()
