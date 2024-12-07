# Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 1. Read the data into a Pandas DataFrame.
# Read the data from the CSV file into a Pandas DataFrame
#df = pd.read_csv('https://static.bc-edx.com/mbc/ai/m5/datasets/bank.csv')
df = pd.read_csv('Random_Forest/data/bank.csv')
# Review the DataFrame
df.head()


# 2. Separate the features `X` from the target `y`
# Separate the features data, X, from the target data, y
y = df['y']
X = df.drop(columns='y')


# 3. Encode the categorical variables from the features data using 
# [`get_dummies`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html).
# Encode the dataset's categorical variables using get_dummies
X = pd.get_dummies(X)

# Review the features DataFrame
X.head()

# 4. Separate the data into training and testing subsets.
# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Review the distinct values from y
y_train.value_counts()

# 5. Scale the data using 
# [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# Instantiate a StandardScaler instance
scaler = StandardScaler()

# Fit the training data to the standard scaler
X_scaler = scaler.fit(X_train)

# Transform the training data using the scaler
X_train_scaled = X_scaler.transform(X_train)

# Transform the testing data using the scaler
X_test_scaled = X_scaler.transform(X_test)

## RandomForestClassifier
# 6. Create and fit a `RandomForestClassifier` to the **scaled** training data.
# Import the RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# Instantiate a RandomForestClassifier instance
model = RandomForestClassifier()

# Fit the traning data to the model
model.fit(X_train_scaled, y_train)

# 7. Make predictions using the scaled testing data.
# Predict labels for original scaled testing features
y_pred = model.predict(X_test_scaled)


## Cluster Centroids
# 8. Import `ClusterCentroids` from `imblearn`.
# Import ClusterCentroids from imblearn
from sklearn.cluster import KMeans
from imblearn.under_sampling import ClusterCentroids

# Instantiate a ClusterCentroids instance
cc_sampler = ClusterCentroids(estimator=KMeans(n_init='auto', random_state=0), random_state=1)

# 9. Fit the `ClusterCentroids` model to the scaled training data.
# Fit the training data to the cluster centroids model
X_resampled, y_resampled = cc_sampler.fit_resample(X_train_scaled, y_train)

# 10. Check the `value_counts` for the resampled target.
# Count distinct values for the resampled target data
y_resampled.value_counts()

# 11. Create and fit a `RandomForestClassifier` to the resampled 
# training data.

# Instantiate a new RandomForestClassier model
cc_model = RandomForestClassifier()

# Fit the resampled data the new model
cc_model.fit(X_resampled, y_resampled)

# 12. Make predictions using the scaled testing data.
# Predict labels for resampled testing features
cc_y_pred = cc_model.predict(X_test_scaled)

# 13. Generate and compare classification reports for each model.
#   * Print a classification report for the model fitted to the original data
#   * Print a classification report for the model fitted to the date resampled with CentroidClusters

# Print classification reports
print(f"Classifiction Report - Original Data")
print(classification_report(y_test, y_pred))
print("---------")
print(f"Classifiction Report - Resampled Data - ClusterCentroids")
print(classification_report(y_test, cc_y_pred))


'''
# Interpretations
Which synthetic resampling tool would you recommend for this application?

ANSWER: SMOTE and SMOTEEN both improved the recall of the "yes" class, 
while Cluster Centroids *dramatically* improved it to almost 100%. 
That said, Cluster Centroids also sacrificed a significant amount of 
precision from the "yes" class and also lost a great deal of overall 
accuracy. It seems that SMOTEEN provides the best improvement of recall 
without sacrificing too much of the other metrics.
'''
