# Import the required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans

# Read the CSV file into a Pandas DataFrame
# Use the date column to create the DateTime Index
df_options = pd.read_csv("https://static.bc-edx.com/mbc/ai/m2/datasets/option-trades.csv", 
                         index_col="date", 
                         parse_dates=True
                        )

# Review the DataFrame
print(df_options.head())


# Create a list for the range of k's to analyze in the elbow plot
# The range should be 1 to 11. 
k = list(range(1, 11))

# Create an empty list to hold inertia scores
inertia = []


# For each k, define and fit a K-means model and append its inertia to the above list
for i in k:
    model = KMeans(n_clusters=i, n_init='auto', random_state=0)
    model.fit(df_options)
    inertia.append(model.inertia_)
    
# View the inertia list
print(inertia)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame from the dictionary holding the values for k and inertia.
df_elbow_data = pd.DataFrame(elbow_data)


# Plot the elbow curve using hvPlot.
df_elbow_data.plot.line(
    x="k", 
    y= "inertia", 
    title="Elbow Curve", 
    xticks=k)
