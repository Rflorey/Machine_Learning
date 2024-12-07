# Initial imports
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Predict Wine Quality with Deep Learning
# Import our input dataset
df  = pd.read_csv('Neural_Network/data/wine_quality.csv')

# Review the DataFrame
print(df .head())

# Create the features (X) and target (y) sets
X = df.drop(columns=["quality"]).values
y = df["quality"].values

# Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create the scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the features data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Creating a Deep Neural Network Model
# Define the model - deep neural net with two hidden layers
number_input_features = 11
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 4

# Create a sequential neural network model
nn = Sequential()

# Add the first hidden layer
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Add the second hidden layer
nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

# Add the output layer
nn.add(Dense(units=1, activation="linear"))

# Compile model
nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

# Fit the model
deep_net_model = nn.fit(X_train_scaled, y_train, epochs=100)

# Defining a new model with four hidden layers
# Define the model - deep neural net with four hidden layers
number_input_features = 11
hidden_nodes_layer1 = 22
hidden_nodes_layer2 = 11
hidden_nodes_layer3 = 8
hidden_nodes_layer4 = 6

# Create a sequential neural network model
nn_2 = Sequential()

# Add the first hidden layer
nn_2.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Add the second hidden layer
nn_2.add(Dense(units=hidden_nodes_layer2, activation="relu"))

# Add the third hidden layer
nn_2.add(Dense(units=hidden_nodes_layer3, activation="relu"))

# Add the fourth hidden layer
nn_2.add(Dense(units=hidden_nodes_layer4, activation="relu"))

# Add the output layer
nn_2.add(Dense(units=1, activation="linear"))

# Compile model
nn_2.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

# Fit the model
deep_net_model_2 = nn_2.fit(X_train_scaled, y_train, epochs=100)


## Evaluating Models Performance
# Evaluate Model 1 using testing data
model1_loss, model1_mse = nn.evaluate(X_test_scaled, y_test, verbose=2)

# Evaluate Model 2 using testing data
model2_loss, model2_mse = nn_2.evaluate(X_test_scaled, y_test, verbose=2)

## Making Predictions
# Make predictions on the testing data
predictions = nn.predict(X_test_scaled).round().astype("int32")

# Create a DataFrame to compare the predictions with the actual values
results = pd.DataFrame({"predictions": predictions.ravel(), "actual": y_test})

# Display sample data
print(results.head(10))

## Saving the Trained Model
# Set the model's file path
file_path = Path("Neural_Network/saved_models/wine_quality.h5")

# Export your model to an HDF5 file
nn.save(file_path)

## Loading a Trained Model
# Import the required libraries
import tensorflow as tf

# Set the model's file path
file_path = Path("Neural_Network/saved_models/wine_quality.h5")

# Load the model to a new object
nn_imported = tf.keras.models.load_model(file_path)

# Evaluate the model using the test data
model_loss, model_accuracy = nn_imported.evaluate(X_test_scaled, y_test, verbose=2)

# Display evaluation results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
