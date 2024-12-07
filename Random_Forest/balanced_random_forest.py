# Import the required modules
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings("ignore")


## Read in the CSV file from the `Resources` folder into a Pandas DataFrame. 
# Read the sba_loans.csv file from the Resources folder into a Pandas DataFrame
loans_df = pd.read_csv('Random_Forest/data/sba_loans.csv')

# Review the DataFrame
print(loans_df.head())

## Create a Series named `y` that contains the data from the "Default" column of the original DataFrame. 
# Note that this Series will contain the labels. Create a new DataFrame named `X` that contains the remaining 
# columns from the original DataFrame. Note that this DataFrame will contain the features.

# Split the data into X (features) and y (labels)

# The y variable should focus on the Default column
y = loans_df['Default']

# The X variable should include all features except the Default column
X = loans_df.drop(columns=['Default'])

### Step 3: Split the features and labels into training and testing sets, and `StandardScaler` your X data.

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale the data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Step 4: Check the magnitude of imbalance in the data set by viewing  the number of distinct values  (`value_counts`) for the labels.
# Count the distinct values in the original labels data
print(y_train.value_counts())


## Step 5: Fit two versions of a random forest model to the data: the first, a regular `RandomForest` classifier, 
# and the second, a `BalancedRandomForest` classifier.
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Making predictions using the testing data
rf_predictions = rf_model.predict(X_test_scaled)

# Import BalancedRandomForestClassifier from imblearn
from imblearn.ensemble import BalancedRandomForestClassifier

# Instantiate a BalancedRandomForestClassifier instance
brf = BalancedRandomForestClassifier()

# Fit the model to the training data
brf.fit(X_train_scaled, y_train)

# Predict labels for testing features
brf_predictions = brf.predict(X_test_scaled)


## Step 6: Resample and fit the training data by one additional method for imbalanced data, 
# such as `RandomOverSampler`, undersampling, or a synthetic technique. Re-esimate by `RandomForest`.
# Import SMOTE from imblearn
from imblearn.over_sampling import SMOTE

# Instantiate the SMOTE model instance
smote_sampler = SMOTE(random_state=1, sampling_strategy='auto')

# Fit the SMOTE model to the training data
X_resampled, y_resampled = smote_sampler.fit_resample(X_train, y_train)

# Fit the RandomForestClassifier on the resampled data
model_resampled_rf = RandomForestClassifier()
model_resampled_rf.fit(X_resampled, y_resampled)

# Generate predictions based on the resampled data model
rf_resampled_predictions = model_resampled_rf.predict(X_test)

## Step 7: Print the confusion matrixes, accuracy scores, and classification reports for the three different models.
# Print the confusion matrix for RandomForest on the original data
confusion_matrix(y_test, rf_predictions)

# Print the confusion matrix for balanced random forest data
confusion_matrix(y_test, brf_predictions)

# Print the confusion matrix for RandomForest on the resampled data
confusion_matrix(y_test, rf_resampled_predictions)


# Print the accuracy score for the original data
baso = balanced_accuracy_score(y_test, rf_predictions)
print(baso)

# Print the accuracy score for the resampled data
basr = balanced_accuracy_score(y_test, brf_predictions)
print(basr)

# Print the accuracy score for the resampled data
basrs = balanced_accuracy_score(y_test, rf_resampled_predictions)
print(basrs)

# Print the classification report for the original data
print(classification_report_imbalanced(y_test, rf_predictions))

# Print the classification report for the resampled data
print(classification_report_imbalanced(y_test, brf_predictions))

# Print the classification report for the resampled data
print(classification_report_imbalanced(y_test, rf_resampled_predictions))


## Step 8: Evaluate the effectiveness of `RandomForest`, `BalancedRandomForest`, and your one additional 
# imbalanced classifier for predicting the minority class. 

### Answer the following question: Does the model generated using one of the imbalanced methods more accurately 
# flag all the loans that eventually defaulted?



# **Question:** Does the model generated using one of the imbalanced methods more accurately flag all the loans 
# that eventually defaulted?
    
# **Answer:** Overall, both resampled models in this example perform better at identifying more of the eventual 
# loan defaults. We can see this by looking at the increase recall for the “default” or “`1`” category in the two 
# imbalanced models, when compared to the  original random forest model.

# A higher recall for this category means that of all the loans that actually were in default, how many did this 
# model correctly catch? A higher recall for a model means it’s going to do a better job at making sure any potential 
# defaults are not missed.

# However, the higher recall for these two imbalanced models comes at a cost: a greater tendency to flag a loan as a 
# potential default, even when it does not. This is evidenced by a lower precision for these two models. If precision 
# looks at, of all those loans the model predicted as default, how many of them actually were defaults, then a lower 
# precision value means that the model is making a lot of false positives; predicting a default when there isn’t actually one. 

# This illustrates the main tradeoff when using imbalanced versions of machine learning models. If you really care about 
# identifying those faulty loans (or whatever you’re trying to predict), and the cost of failing to identify a faulty loan 
# is very high, then maybe an imbalanced model with lower precision is worth it. After all, we can always find another 
# business to lend to, but if that business defaults, it’s very costly to us as a lender. 

# If on the other hand, you have a situation in which the costs of misclassification are the same either way—if failing to 
# correctly identify a `1` has the same practical cost as failing to correctly identify a `0`—then we may be better off 
# with the overall higher accuracy of a standard machine learning model.