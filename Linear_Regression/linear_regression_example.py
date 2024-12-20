# Import required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## Load and Visualize the Sales Data
# Read the sales data
file_path = "https://static.bc-edx.com/mbc/ai/m3/datasets/sales.csv"
df_sales = pd.read_csv(file_path)

# Display sample data
print(df_sales.head())

# Create a scatter plot with the sales information
sales_plot = df_sales.plot.scatter(
    x="ads",
    y="sales",
    title="Sales per Number of Ads"
)
 

## Prepare the Data to Fit the Linear Regression Model
# Create the X set by using the `reshape` function to format the ads data as a single column array.
X = df_sales["ads"].values.reshape(-1, 1)

# Display sample data
print(X[:5])

# Create an array for the dependent variable y with the sales data
y = df_sales["sales"]

## Build the Linear Regression Model
# Create a model with scikit-learn
model = LinearRegression()

# Fit the data into the model
model.fit(X, y)

# Display the slope
print(f"Model's slope: {model.coef_}")

# Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Display the model's best fit line formula
print(f"Model's formula: y = {model.intercept_} + {model.coef_[0]}X")


## Plot the Best Fit Line for the Sales Prediction Model
# Make predictions using the X set
predicted_y_values = model.predict(X)

# Create a copy of the original data
df_sales_predicted = df_sales.copy()

# Add a column with the predicted sales values
df_sales_predicted["sales_predicted"] = predicted_y_values

# Display sample data
print(df_sales_predicted.head())


# Create a line plot of the predicted salary values
best_fit_line = df_sales_predicted.plot.line(
    x = "ads",
    y = "sales_predicted",
    color = "red"
)



# Superpose the original data and the best fit line
# Create a scatter plot with the sales information
sales_plot = df_sales_predicted.plot.scatter(
    x="ads",
    y="sales",
    title="Sales per Number of Ads"
)

best_fit_line = df_sales_predicted.plot.line(
    x = "ads",
    y = "sales_predicted",
    color = "red",
    ax=sales_plot
)

plt.show() 
## Make Manual Predictions
# Display the formula to predict the sales with 100 ads
print(f"Model's formula: y = {model.intercept_} + {model.coef_[0]} * 100")

# Predict the sales with 100 ads
y_100 = model.intercept_ + model.coef_[0] * 100

# Display the prediction
print(f"Predicted sales with 100 ads: ${y_100:.2f}")

## Make Predictions Using the `predict` Function
# Create an array to predict sales for 100, 150, 200, 250, and 300 ads
X_ads = np.array([100, 150, 200, 250, 300])

# Format the array as a one-column array
X_ads = X_ads.reshape(-1,1)

# Display sample data
print(X_ads)

# Predict sales for 100, 150, 200, 250, and 300 ads
predicted_sales = model.predict(X_ads)

# Create a DataFrame for the predicted sales
df_predicted_sales = pd.DataFrame(
    {
        "ads": X_ads.reshape(1, -1)[0],
        "predicted_sales": predicted_sales
    }
)

# Display data
print(df_predicted_sales)


## Linear Regression Model Assessment
# Import relevant metrics - score, r2, mse, rmse - from Scikit-learn
from sklearn.metrics import mean_squared_error, r2_score

# Compute the metrics for the linear regression model
score = round(model.score(X, y, sample_weight=None),5)
r2 = round(r2_score(y, predicted_y_values),5)
mse = round(mean_squared_error(y, predicted_y_values),4)
rmse = round(np.sqrt(mse),4)

# Print relevant metrics.
print(f"The score is {score}.")
print(f"The r2 is {r2}.")
print(f"The mean squared error is {mse}.")
print(f"The root mean squared error is {rmse}.")