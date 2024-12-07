# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
# Preventing Credit Card Defaults

In this activity, you will use Keras to build a neural network model 
that predicts whether a credit card customer will default on their debt.
'''

## Step 1: Read the CSV file into a Pandas DataFrame.
# Read the cc_default.csv file into a Pandas DataFrame
cc_df = pd.read_csv('Neural_Network/data/cc_default.csv')

# Review the DataFrame
cc_df.head()

## Step 2: Define the features set `X` by including all of the DataFrame columns except the “DEFAULT” column.

# Define features set X by selecting all columns but DEFAULT
X = cc_df.drop(columns=["DEFAULT"]).copy()

# Display the features DataFrame
X.head()

## Step 3: Create the target `y` by assigning the values of the DataFrame “DEFAULT” column.

# Define target set by selecting the DEFAULT column
y = cc_df["DEFAULT"]

# Display a sample y
y[:5]

## Step 4: Create the training and testing sets using the `train_test_split` function from scikit-learn.

# Create training and testing datasets using train_test_split
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

## Step 5: Scale the features data using the `StandardScaler` from sklearn.
# Create the StandardScaler instance
X_scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_scaler.fit(X_train)

# Scale both the training and testing data from the features dataset
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Step 6: Create a neural network model with an input layer that consists of 
# 22 inputs, one hidden layer, and an output layer. Use the `units` parameter 
# to define 12 neurons for the hidden layer and a single output for the output 
# layer. Use the ReLU activation function for the hidden layer and the sigmoid 
# activation function for the output layer.

# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define the the number of inputs to the model
number_inputs = 22

# Define the number of hidden nodes for the model
number_hidden_nodes = 12

# Create the Sequential model instance
neuron = Sequential()

# Add a Dense layer specifying the number of inputs, the number of hidden nodes, and the activation function
neuron.add(Dense(units=number_hidden_nodes, input_dim=number_inputs, activation="relu"))

# Add the output layer to the model specifying the number of output neurons and activation function
neuron.add(Dense(1, activation="sigmoid"))

## Step 7: Display the model structure using the `summary` function.
# Display the Sequential model summary
neuron.summary()


## Step 8:  Compile the neural network model using the `binary_crossentropy` 
# loss function, the `adam` optimizer, and the additional metric `accuracy`.

# Compile the Sequential model
neuron.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

## Step 9: Fit the model with 100 epochs using the training data.

# Fit the model using 100 epochs and the training data
model = neuron.fit(X_train_scaled, y_train, epochs=100)


## Step 10: Plot the model’s loss function and accuracy over the 100 epochs.
# Create a DataFrame using the model history and an index parameter
model_plot = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))

# Visualize the model plot where the y-axis displays the loss metric
model_plot.plot(y="loss")

# Visualize the model plot where the y-axis displays the accuracy metric
model_plot.plot(y="accuracy")

## Step 11: Evaluate the model using testing data and the `evaluate` method.
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = neuron.evaluate(X_test_scaled, y_test, verbose=2)

# Display the evaluation results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

## Step 12: Use the model and your scaled `X` test data to predict the `y` values for the test data.
# Make predictions on test data
predictions = (neuron.predict(X_test_scaled) > 0.5).astype("int32")

## Step 13: Create a dataframe that includes the predicted `y` values and the actual `y` values. 
# Display a sample of this dataframe.
# Create a DataFrame to compare the predictions with the actual values
results = pd.DataFrame({"predictions": predictions.ravel(), "actual": y_test})

# Display sample data
print(results.head(20))