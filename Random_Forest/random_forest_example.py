# Initial imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Loading and Preprocessing Malware Apps Data
# Load the `app_data.csv` in a pandas DataFrame called `df_apps`

# Load data
df_apps = pd.read_csv('Random_Forest/data/app_data.csv')
df_apps.head()

#Define the features set, by copying the `df_apps` DataFrame and 
# dropping the `Result` column.
# Define features set
X = df_apps.copy()
X.drop("Result", axis=1, inplace=True)
X.head()

#Create the target vector by assigning the values of the `Result` 
# column from the `df_apps` DataFrame.

# Define target set
y = df_apps["Result"].ravel()
y[:5]

#Split the data into training and testing sets.
# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

## Fitting the Random Forest Model

#Create a random forest instance and train it with the training data 
# (`X_train` and `y_train`), define `n_estimators=128` and 
# `random_state=78`.

# Create the random forest classifier instance
rf_model = RandomForestClassifier(n_estimators=128, random_state=78)

# Fit the model
rf_model = rf_model.fit(X_train, y_train)

## Making Predictions Using the Random Forest Model
# Validate the trained model by malware apps using the testing 
# data (`X_test`).

# Make predictions using the testing data
predictions = rf_model.predict(X_test)

## Model Evaluation
# Evaluate model's results, by using `sklearn` to calculate 
# the accuracy score.

# Calculate the accuracy score
acc_score = accuracy_score(y_test, predictions)

# Display results
print(f"Accuracy Score : {acc_score}")

# Feature Importance
# In this section, you are asked to fetch the features' importance 
# from the random forest model and display the top 10 most important 
# features.

# Get the feature importance array
importances = rf_model.feature_importances_
# List the top 10 most important features
importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
importances_sorted[:10]

