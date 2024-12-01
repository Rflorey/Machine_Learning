# Import the required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans

 
# Read in the CSV file and create the Pandas DataFrame
df_shopping = pd.read_csv("https://static.bc-edx.com/mbc/ai/m2/datasets/shopping-data.csv")
# Review the DataFrame
print(df_shopping.head())

# Check the DataFrame data types
print(df_shopping.dtypes)

# Build the encodeCard helper function

# Credit card purchases should encode to 1
# Debit card purchases should encode to 0
def encodeCard(card_type):
    """
    This function encodes a card type by setting credit card purchases to 1 and debit cards to 0.
    """
    if card_type.lower() == "credit":
        return 1
    else:
        return 0
    
# Edit the `Card Type` column using the encodeCard function
df_shopping["Card Type"] = df_shopping["Card Type"].apply(encodeCard)

# Review the DataFrame
print(df_shopping.head())

# Scale the Annual Income column 
df_shopping["Annual Income"] = df_shopping["Annual Income"] / 1000

# Review the DataFrame
print(df_shopping.head())

# Drop the CustomerID column
df_shopping = df_shopping.drop(columns=["CustomerID"])

# Review the DataFrame
print(df_shopping.head())
print(df_shopping.tail())

# Initialize the K-Means model; n_clusters=4 and n_init='auto'
model_k4 = KMeans(n_clusters=4, n_init='auto')

# Fit the model
model_k4.fit(df_shopping)

# Predict the model segments (clusters)
customer_segments_k4 = model_k4.predict(df_shopping)

# View the customer segments
print(customer_segments_k4)


# Initialize the K-Means model; n_clusters=5 and n_init='auto'
model_k5 = KMeans(n_clusters=5, n_init='auto')


# Fit the model
model_k5.fit(df_shopping)


# Predict the model segments (clusters)
customer_segments_k5 = model_k5.predict(df_shopping)

# View the customer segments
print(customer_segments_k5)

# Crate a copy of the df_shopping DataFrame and name it as df_shopping_predictions
df_shopping_predictions = df_shopping.copy()

# Create a new column in the DataFrame with the predicted clusters with k=4
df_shopping_predictions["Customer Segment (k=4)"] = customer_segments_k4


# Create a new column in the DataFrame with the predicted clusters with k=5
df_shopping_predictions["Customer Segment (k=5)"] = customer_segments_k5

# Review the DataFrame
print(df_shopping_predictions.head())

# Create a scatter plot with with x="Annual Income" and y="Spending Score (1-100)" with k=4 segments
df_shopping_predictions.plot.scatter(
    x="Annual Income", 
    y="Spending Score", 
    c="Customer Segment (k=4)",
    title = "Scatter Plot by Stock Segment - k=4",
    colormap='winter'
)

# Create a scatter plot with x="Annual Income" and y="Spending Scoree" with k=5 segments
df_shopping_predictions.plot.scatter(
    x="Annual Income", 
    y="Spending Score", 
    c="Customer Segment (k=5)",
    title = "Scatter Plot by Stock Segment - k=5",
    colormap='winter'
)
    