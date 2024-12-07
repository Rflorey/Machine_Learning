# Load libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Import data and plot
#df = pd.read_csv('https://static.bc-edx.com/mbc/ai/m5/datasets/spirals.csv')
df = pd.read_csv('SVM/data/spirals.csv')
df.plot.scatter("x1", "x2", color=df["y"].map({1.0: "purple", 0.0:"orange"})).get_figure().savefig("SVM/data/spirals_data_scatter.png")
# plt.show()

# Split the data into training and testing sets
X = df[['x1', 'x2']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)

# Train and score a model
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_pred, y_test))

# Plot the predictions made by the model
X_test.plot.scatter("x1", 
                    "x2",
                    color = pd.Series(y_pred)\
                    .map({1.0: "purple", 0.0:"orange"})).get_figure().savefig("SVM/data/spirals_predictions_scatter.png")

# Interpretations
'''
The SVC model was not able to fully learn the spiral shape of the data. 
This is a somewhat forced example; you have already learned to use models 
that could classify the points in this spirals dataset easily (RandomForest, for instance). 
However, just as this dataset stumps SVC, there are more complex datasets that will stump 
RandomForest. This issue of overcoming increasingly complex data is where huge strides have 
been made in Machine Learning in recent years; Neural Networks are some of the most powerful 
algorithms in use today!
'''