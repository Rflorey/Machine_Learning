# Linear Regression Sales Analysis

A Python script that performs linear regression analysis on advertising and sales data to predict sales based on advertising spend.

## Overview

This script demonstrates how to build and evaluate a simple linear regression model using scikit-learn. It analyzes the relationship between advertising expenditure and sales, providing predictions and model evaluation metrics.

## Features

- Data loading and visualization of sales vs. advertising relationship
- Linear regression model implementation
- Sales predictions for various advertising scenarios
- Model performance evaluation using multiple metrics
- Visualization of the regression line with actual data points

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install required packages using:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. Load the script and ensure the sales data CSV file is accessible
2. Run the script to:
   - Visualize the sales data
   - Build the linear regression model
   - Make predictions
   - Evaluate model performance

## Input Data Format

The script expects a CSV file with at least two columns:
- `ads`: Number of advertisements
- `sales`: Corresponding sales values

## Output

The script provides:
- Scatter plot of actual sales data
- Linear regression line overlaid on the data
- Predicted sales for specific advertising values
- Model evaluation metrics:
  - R-squared score
  - Mean squared error (MSE)
  - Root mean squared error (RMSE)

## Model Metrics

The model evaluates performance using:
- Model score (R-squared)
- Mean squared error
- Root mean squared error

## Example Usage

```python
# Load the data
df_sales = pd.read_csv("sales.csv")

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predicted_sales = model.predict(X_new)
```

## License

This project is available under the MIT License.