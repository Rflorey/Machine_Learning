import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv("Logistic_Regression/data/blobs_data.csv")
df.head()


# Visualize the data
'''
Data can be difficult to understand without a visual. We'll plot the data here to 
help us understand the goal of the model. We'll use the "X" data (both "X1" and "X2") 
to position the points and we'll color the points using the "y" data. Every positive 
row (with a 1 in the "y" column) will be colored orange. Every negative row (with a 0 
in the "y" column) will be colored purple. After training, our model should be able 
to predict the correct "color" of each point using just it's position on the chart. 
'''

df.plot.scatter(x="X1", y="X2", color=df['y'].map({1: "orange", 0: "purple"}), alpha=0.8)

'''
# Separate features and target
The "y" variable will hold the values we'll eventually try to predict. The "X" 
variable will hold all the values we can use to make our prediction. Then we'll 
split the data into training and testing sets.
'''
X = df[["X1", "X2"]]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)

'''
# Create the model
We'll create the model, fit the model using the training data, and use the 
model to make predictions on the testing data.
'''

# Create a Logistic Regression Model
classifier = LogisticRegression(solver='lbfgs', random_state=1)

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Calculate the accuracy of the model
classifier.score(X_test, y_test)

'''
 # Confusion Matrix
 The confusion matrix will allow us to see how our predictions break down 
 by class.
'''

# Create a confusion matrix
confusion_matrix(y_test, predictions, labels = [1,0])


'''
# Classification Report

Add your code to the cell below to create a classification report that shows 
precision, recall, and f1-score for both the purple and orange classes.
'''

from sklearn.metrics import classification_report

target_names = ["Class Purple", "Class Orange"]
print(classification_report(y_test, predictions, target_names=target_names))

# Interpretation
'''
The classification report shows that our model is MUCH better at predicting purple 
data points than orange data points. What do the precision and recall show for both 
orange and purple?

ANSWER: Precision and recall are perfectly balanced for purple data points, but the 
model has more precision than recall for orange data points. The model is also 
significantly better at predicting purple points than orange points.
'''