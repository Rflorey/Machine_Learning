import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import the data
df = pd.read_csv("ML_Optimization/data/blobs_data.csv")
print(df.head())

# Visualize the data
'''
# Visualize the data
Data can be difficult to understand without a visual. 
plot the data here to helps understand the goal of the model. 
The "X" data (both "X1" and "X2") positions the points and colors the points using the "y" data. 
Every positive row (with a 1 in the "y" column) will be colored orange. Every negative row (with a 0 in the "y" column) will be 
colored purple. After training, our model should be able to predict the correct "color" of each point using just it's position 
on the chart. 
'''
df.plot.scatter(x="X1", y="X2", color=df['y'].map({1: "orange", 0: "purple"}), alpha=0.8)

# Separate features and target
'''
The "y" variable will hold the values we'll eventually try to predict. The "X" variable will hold all the values we can 
use to make our prediction. Then we'll split the data into training and testing sets.

NOTE:
The 'lbfgs' solver is the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm. 
It's an optimization algorithm used in Logistic Regression that belongs to the family of quasi-Newton methods.

The 'lbfgs' solver is a good default choice for most logistic regression problems, 
especially when dealing with moderate-sized datasets and when using L2 regularization.
'''
X = df[["X1", "X2"]]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)

## Create the model
# Create a Logistic Regression Model
classifier = LogisticRegression(solver='lbfgs', random_state=1)

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Calculate the accuracy of the model
print(f"Testing Accuracy Score: {classifier.score(X_test, y_test)}")

# Confusion Matrix
# The confusion matrix will allow us to see how our predictions break down by class.
# Create a confusion matrix
con_mat =confusion_matrix(y_test, predictions, labels = [1,0])

# Print header
print("\nConfusion Matrix")
print("-" * 40)
# Print column headers
print(f"{'':>15}Predicted Pos  Predicted Neg")
 # Print matrix with row labels
print(f"Actual Pos{con_mat[0][0]:>13}{con_mat[0][1]:>14}")
print(f"Actual Neg{con_mat[1][0]:>13}{con_mat[1][1]:>14}")
print("-" * 40)
plt.show()

'''
Matrix Breakdown:
True Positives (TP) = 14 (correctly identified positives)
False Negatives (FN) = 11 (incorrectly classified as negative)
False Positives (FP) = 10 (incorrectly classified as positive)
True Negatives (TN) = 240 (correctly identified negatives)

Key Metrics:
Accuracy = (TP + TN) / Total = (14 + 240) / 275 ≈ 92.4%
Precision = TP / (TP + FP) = 14 / 24 ≈ 58.3%
Recall (Sensitivity) = TP / (TP + FN) = 14 / 25 = 56%
Specificity = TN / (TN + FP) = 240 / 250 = 96%

Interpretation:
The model is very good at identifying negative cases (96% specificity)
There's a class imbalance in the dataset (25 positive cases vs 250 negative cases)
The model has moderate precision and recall for positive cases
While the overall accuracy is high (92.4%), this is largely due to the model's strong performance on the majority class (negative cases)
The model struggles more with identifying positive cases correctly, as shown by the lower precision and recall values
'''
