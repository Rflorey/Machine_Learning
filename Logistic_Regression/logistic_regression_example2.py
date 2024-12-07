import pandas as pd
import matplotlib.pyplot as plt

# Prepare the Data

# Import the data
df = pd.read_csv('Logistic_Regression/data/start_up_success.csv')



# Plot the data on a scatter plot
df.plot.scatter(
    x='Industry Health', 
    y='Financial Performance', 
    c='Firm Category', 
    marker='o', 
    s=25, 
    edgecolor='k',
    colormap="winter"
)
plt.show()
# Preview the DataFrame
print(df.head(3))

# Check the number of unhealthy vs. healthy firms ('Firm Category')
# using value_counts
df['Firm Category'].value_counts()

# Split the data into training and testing sets
# Import Module
from sklearn.model_selection import train_test_split

# Split training and testing sets
# Create the features DataFrame, X
X = df.copy()
X = X.drop(columns='Firm Category')

# Create the target DataFrame, y
y = df['Firm Category']

# Use train_test_split to separate the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Review the X_train DataFrame
print(X_train)


# Model and Fit the Data to a Logistic Regression
# Import `LogisticRegression` from sklearn
from sklearn.linear_model import LogisticRegression

# Create a `LogisticRegression` function and assign it 
# to a variable named `logistic_regression_model`.
logistic_regression_model = LogisticRegression()

# Fit the model
logistic_regression_model.fit(X_train, y_train)

# Score the model
print(f"Training Data Score: {logistic_regression_model.score(X_train, y_train)}")
print(f"Testing Data Score: {logistic_regression_model.score(X_test, y_test)}")

# Generate predictions from the model we just fit
predictions = logistic_regression_model.predict(X_train)

# Convert those predictions (and actual values) to a DataFrame
results_df = pd.DataFrame({"Prediction": predictions, "Actual": y_train})
print(results_df)


# Apply the fitted model to the `test` dataset
testing_predictions = logistic_regression_model.predict(X_test)

# Save both the test predictions and actual test values to a DataFrame
results_df = pd.DataFrame({
    "Testing Data Predictions": testing_predictions, 
    "Testing Data Actual Targets": y_test})

# Display the results DataFrame
print(results_df)


# Import the accuracy_score function
from sklearn.metrics import accuracy_score

# Calculate the model's accuracy on the test dataset
accuracy_score(y_test, testing_predictions)

