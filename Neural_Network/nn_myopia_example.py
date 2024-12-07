from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf

'''
# Predicting Myopia
In this activity, you'll use a deep learning model to predict whether 
a person has myopia.
'''
# Import our input dataset
myopia_df = pd.read_csv('Neural_Network/data/myopia.csv')

# Review the DataFrame
print(myopia_df.head())

# Create the features and target sets
y = myopia_df["MYOPIC"].values
X = myopia_df.drop(columns="MYOPIC").copy()

# Preprocess numerical data for neural network
# Create a StandardScaler instances
scaler = StandardScaler()

# Scale the data
X = scaler.fit_transform(X)


# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# Define the deep learning model 
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=16, activation="relu", input_dim=14))
nn_model.add(tf.keras.layers.Dense(units=16, activation="relu"))
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the Sequential model together and customize metrics
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn_model.fit(X_train, y_train, epochs=50)

# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Make predictions on the test data
predictions = (nn_model.predict(X_test) > 0.5).astype("int32")

# Create a DataFrame to compare the predictions with the actual values
results = pd.DataFrame({"predictions": predictions.ravel(), "actual": y_test})

# Display sample data
print(results.head(10))



