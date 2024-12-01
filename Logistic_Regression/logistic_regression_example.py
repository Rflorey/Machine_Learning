# Predicting Malware

## Reference:
# Mathur,Akshay & Mathur,Akshay. (2022). [NATICUSdroid (Android Permissions) Dataset](https://archive-beta.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset). UCI Machine Learning Repository.

# Import the required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

## Prepare the Data
# Read in the app-data.csv file into a Pandas DataFrame.
file_path = "https://static.bc-edx.com/mbc/ai/m4/datasets/app-data.csv"
app_data = pd.read_csv(file_path)

# Review the DataFrame
print(app_data.head())

# The column 'Result' is the predict. 
# Class 0 indicates a benign app and class 1 indicates a malware app
# Using value_counts, how many malware apps are in this dataset?
print(app_data["Result"].value_counts())

## Split the data into training and testing sets
# The target column `y` should be the binary `Result` column.
y = app_data["Result"]

# The `X` should be all of the features. 
X = app_data.copy()
X = X.drop(columns="Result")

# Split the dataset using the train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y)


## Model and Fit the Data to a Logistic Regression
# Declare a logistic regression model.
# Apply a random_state of 7 and max_iter of 120 to the model
logistic_regression_model = LogisticRegression(random_state=7, max_iter=120)

# Fit and save the logistic regression model using the training data
lr_model = logistic_regression_model.fit(X_train, y_train)

# Validate the model
print(f"Training Data Score: {lr_model.score(X_train, y_train)}")
print(f"Testing Data Score: {lr_model.score(X_test, y_test)}")

## Predict the Testing Labels
# Make and save testing predictions with the saved logistic regression model using the test data
testing_predections = lr_model.predict(X_test)

# Review the predictions
# print(testing_predections)

## Calculate the Performance Metrics
# Display the accuracy score for the test dataset.
# print(accuracy_score(y_test, testing_predections))
