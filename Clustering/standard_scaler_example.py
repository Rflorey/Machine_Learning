# Import the dependencies.
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data into a Pandas DataFrame
df_shopping = pd.read_csv("https://static.bc-edx.com/mbc/ai/m2/datasets/shopping-data.csv")

# Drop the CustomerID column
df_shopping = df_shopping.drop(columns=["CustomerID"])

# Display sample data
print(df_shopping.head())

# Scaling the numeric columns
shopping_data_scaled = StandardScaler().fit_transform(df_shopping[["Age", "Annual Income", "Spending Score"]])

# Creating a DataFrame with with the scaled data
df_shopping_transformed = pd.DataFrame(shopping_data_scaled, columns=["Age", "Annual Income", "Spending Score"])

# Display sample data
print(df_shopping_transformed.head())

# Transform the Card Type column using get_dummies()
card_dummies = pd.get_dummies(df_shopping["Card Type"]).astype(int)

# Display sample data
print(card_dummies.head())

# Concatenate the df_shopping_transformed and the card_dummies DataFrames
df_shopping_transformed = pd.concat([df_shopping_transformed, card_dummies], axis=1)

# Display sample data
print(df_shopping_transformed.head())