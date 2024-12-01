# Required imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

## Load the Data Into a Pandas DataFrame
# Read in the CSV file as a Pandas Dataframe
ccinfo_default_df = pd.read_csv("https://static.bc-edx.com/mbc/ai/m2/datasets/ccinfo-transformed.csv")

print(ccinfo_default_df.head())

# Plot the clusters using the "limit_bal" and "age" columns
ccinfo_default_df.plot.scatter(
    x="limit_bal",
    y="age",
    c="customer_segments",
    colormap="winter")
# plt.show()

# Plot the clusters using the "bill_amt" and "pay_amt" columns
ccinfo_default_df.plot.scatter(
    x="bill_amt",
    y="pay_amt",
    c="customer_segments",
    colormap="winter")

## Use PCA to reduce the number of factors 
# Import the PCA module
from sklearn.decomposition import PCA

# Instantiate the PCA instance and declare the number of PCA variables
pca = PCA(n_components=2)

## Creating the PCA DataFrame
# Fit the PCA model on the transformed credit card DataFrame
ccinfo_pca = pca.fit_transform(ccinfo_default_df)

# Review the first 5 rows of list data
print(ccinfo_pca[:5])


##PCA explained variance ratio
# Calculate the PCA explained variance ratio
pca.explained_variance_ratio_


# Create the PCA DataFrame
ccinfo_pca_df = pd.DataFrame(
    ccinfo_pca,
    columns=["PCA1", "PCA2"]
)

# Review the PCA DataFrame
print(ccinfo_pca_df.head())


## Determine the PCA Components Feature importance and variance contribution
# Get feature names from original dataframe
feature_names = ccinfo_default_df.columns.tolist()

# Create a DataFrame of feature loadings
loadings = pd.DataFrame(
    pca.components_.T,    # Transpose the components matrix
    columns=['PC1', 'PC2'],
    index=feature_names
)

# Add absolute values for easier ranking
loadings['PC1_abs'] = abs(loadings['PC1'])
loadings['PC2_abs'] = abs(loadings['PC2'])

# Sort features by importance for each component
print("\nTop features for PC1 (sorted by absolute value):")
print(loadings.sort_values('PC1_abs', ascending=False)[['PC1']])

print("\nTop features for PC2 (sorted by absolute value):")
print(loadings.sort_values('PC2_abs', ascending=False)[['PC2']])

# Calculate and print explained variance ratio as percentage
explained_variance = pca.explained_variance_ratio_ * 100
print("\nExplained variance ratio:")
print(f"PC1: {explained_variance[0]:.2f}%")
print(f"PC2: {explained_variance[1]:.2f}%\n")

#  Create a more visual representation with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loadings[['PC1', 'PC2']], annot=True, cmap='coolwarm', center=0)
plt.title('PCA Components Feature Loadings')
plt.tight_layout()



## Incorporating the PCA DataFrame into the elbow method
# Create a a list to store inertia values and the values of k
inertia = []
k = list(range(1, 11))

# Append the value of the computed inertia from the `inertia_` attribute of teh KMeans model instance
for i in k:
    k_model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    k_model.fit(ccinfo_pca_df)
    inertia.append(k_model.inertia_)

# Define a DataFrame to hold the values for k and the corresponding inertia
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)

# Review the DataFrame
print(df_elbow.head(10))
print(" ")
# Plot the Elbow Curve
df_elbow.plot.line(
    x="k", 
    y="inertia"
)

# Determine the rate of decrease between each k value. 
k = df_elbow["k"]
inertia = df_elbow["inertia"]
for i in range(1, len(k)):
    percentage_decrease = (inertia[i-1] - inertia[i]) / inertia[i-1] * 100
    print(f"Percentage decrease from k={k[i-1]} to k={k[i]}: {percentage_decrease:.2f}%")



## Segmentation of the PCA data with K-means 
# Define the model with 3 clusters
model = KMeans(n_clusters=3, n_init='auto', random_state=0)

# Fit the model
model.fit(ccinfo_pca_df)

# Make predictions
k_3 = model.predict(ccinfo_pca_df)

# Create a copy of the PCA DataFrame
ccinfo_pca_predictions_df = ccinfo_pca_df.copy()

# Add a class column with the labels
ccinfo_pca_predictions_df["customer_segments"] = k_3    


# Plot the clusters
ccinfo_pca_predictions_df.plot.scatter(
    x="PCA1",
    y="PCA2",
    c="customer_segments",
    colormap="winter"
)
plt.show()