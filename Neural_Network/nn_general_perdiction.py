# Initial imports
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


## Data Generation and Preprocessing
# Generate 1000 demo data samples with 2 features and two centers
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)

# Transforming y to a vertical vector
y = y.reshape(-1, 1)
y.shape

# Creating a DataFrame with the dummy data
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y
print(df.head())

# Plotting the dummy data
df.plot.scatter(x="Feature 1", y="Feature 2", c="Target", colormap="winter")

# Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Creating a Neural Network Model
# Create a sequential neural network model
neuron = Sequential()

# Add the input and the hidden layer to the model
number_inputs = 2
number_hidden_nodes = 1

neuron.add(Dense(units=number_hidden_nodes, activation="relu", input_dim=number_inputs))

# Add the output layer
number_classes = 1

neuron.add(Dense(units=number_classes, activation="sigmoid"))

# Display model summary
print(neuron.summary())


## Compiling a Neural Network Model
# Compile the model
neuron.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fitting the model
model = neuron.fit(X_train_scaled, y_train, epochs=100)

# Create a DataFrame with the history dictionary
df = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))

# Plot the loss
df.plot(y="loss")

# Plot the accuracy
df.plot(y="accuracy")

plt.show()### Evaluating the Model Performance
# Evaluate the model using testing data
model_loss, model_accuracy = neuron.evaluate(X_test_scaled, y_test, verbose=2)

# Display evaluation results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

## Making Predictions with a Neural Network Model
# Create 10 new samples of dummy data
new_X, new_y = make_blobs(n_samples=10, centers=2, n_features=2, random_state=1)

# Making predictions
predictions = (neuron.predict(new_X) > 0.5).astype("int32")

# Create a DataFrame to compare the predictions with the actual values
results = pd.DataFrame({"predictions": predictions.ravel(), "actual": new_y})

# Display sample data
print(results.head(10))
