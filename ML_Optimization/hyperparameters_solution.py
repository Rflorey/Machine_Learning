import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#df = pd.read_csv('https://static.bc-edx.com/mbc/ai/m5/datasets/numeric_bank.csv')
df = pd.read_csv('ML_Optimization/data/numeric_bank.csv')
df.head()


target = df["y"]
target_names = ["negative", "positive"]


data = df.drop("y", axis=1)
data.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)


# Create three KNN classifiers
from sklearn.neighbors import KNeighborsClassifier
untuned_model = KNeighborsClassifier()
grid_tuned_model = KNeighborsClassifier()
random_tuned_model = KNeighborsClassifier()


## Train a model without tuning
from sklearn.metrics import classification_report
untuned_model.fit(X_train, y_train)
untuned_y_pred = untuned_model.predict(X_test)
print(classification_report(y_test, untuned_y_pred,
                            target_names=target_names))


# Create the grid search estimator along with a parameter object containing the values to adjust.
# Try adjusting n_neighbors with values of 1 through 19. Adjust leaf_size by using 10, 50, 100, and 500.
# Include both uniform and distance options for weights.
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['uniform', 'distance'],
    'leaf_size': [10, 50, 100, 500]
}
grid_clf = GridSearchCV(grid_tuned_model, param_grid, verbose=3)


# Fit the model by using the grid search estimator.
# This will take the KNN model and try each combination of parameters.
grid_clf.fit(X_train, y_train)


# List the best parameters for this dataset
print(grid_clf.best_params_)

# Print the classification report for the best model
grid_y_pred = grid_clf.predict(X_test)
print(classification_report(y_test, grid_y_pred,
                            target_names=target_names))


# Create the parameter object for the randomized search estimator.
# Try adjusting n_neighbors with values of 1 through 19. 
# Adjust leaf_size by using a range from 1 to 500.
# Include both uniform and distance options for weights.
param_grid = {
    'n_neighbors': np.arange(1,20,2),
    'weights': ['uniform', 'distance'],
    'leaf_size': np.arange(1, 500)
}
param_grid

# Create the randomized search estimator
from sklearn.model_selection import RandomizedSearchCV
random_clf = RandomizedSearchCV(random_tuned_model, param_grid, random_state=0, verbose=3)


# Fit the model by using the randomized search estimator.
random_clf.fit(X_train, y_train)

# List the best parameters for this dataset
print(random_clf.best_params_)


# Make predictions with the hypertuned model
random_tuned_pred = random_clf.predict(X_test)


# Calculate the classification report
print(classification_report(y_test, random_tuned_pred,
                            target_names=target_names))
