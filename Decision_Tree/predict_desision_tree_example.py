# Initial imports
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Needed for decision tree visualization
import pydotplus
from IPython.display import Image

## Loading and Preprocessing Crowdfunding Data
'''
Load the `crowdfunding-data.csv` in a pandas DataFrame called 
`df_crowdfunding`.
'''

# Load data
df_crowdfunding = pd.read_csv('Decision_Tree/data/crowdfunding_data.csv')
print(df_crowdfunding.head())

# Define the features set, by copying the `df_crowdfunding` 
# DataFrame and dropping the `outcome` column.
# Define features set
X = df_crowdfunding.copy()
X.drop("outcome", axis=1, inplace=True)
X.head()

# Create the target vector by assigning the values of the `outcome` 
# column from the `df_crowdfunding` DataFrame.

# Define target vector
y = df_crowdfunding["outcome"].values.reshape(-1, 1)
y[:5]

# Split the data into training and testing sets.
# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Use the `StandardScaler` to scale the features data, remember that only `X_train` and `X_testing` DataFrames should be scaled.
# Create the StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler with the training data
X_scaler = scaler.fit(X_train)

# Scale the training data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Fitting the Decision Tree Model
#Once data is scaled, create a decision tree instance and train it 
# with the training data (`X_train_scaled` and `y_train`).

# Create the decision tree classifier instance
model = tree.DecisionTreeClassifier()

# Fit the model
model = model.fit(X_train_scaled, y_train)


## Making Predictions Using the Tree Model
#Validate the trained model, by predicting malware apps using the testing data (`X_test_scaled`).
# Making predictions using the testing data
predictions = model.predict(X_test_scaled)


## Model Evaluation
#Evaluate model's results, by using `sklearn` to calculate the accuracy score.
# Calculate the accuracy score
acc_score = accuracy_score(y_test, predictions)

print(f"Accuracy Score : {acc_score}")

'''
## Visualizing the Decision Tree
In this section, you should create a visual representation of the 
decision tree using `pydotplus`. Show graph, and also save it 
in `PDF` and `PNG` formats.
'''
# Create DOT data
dot_data = tree.export_graphviz(
    model, out_file=None, feature_names=X.columns, class_names=["0", "1"], filled=True
)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())


# When saving the image, graph.write_<file_type>() must take a string object

# Save the tree as PDF
file_path = "Decision_Tree/graphs/crowdfunding_tree.pdf"
graph.write_pdf(file_path)

# Save the tree as PNG
file_path = "Decision_Tree/graphs/crowdfunding_tree.png"
graph.write_png(file_path)

