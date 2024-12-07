# Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 1. Read the CSV file into a Pandas DataFrame
# Read the CSV file into a Pandas DataFrame
#df = pd.read_csv('https://static.bc-edx.com/mbc/ai/m5/datasets/bank.csv')
df = pd.read_csv('Random_Forest/data/bank.csv')

# Review the DataFrame
df.head()

# 2. Separate the features, `X`, from the target, `y`, data.
# Split the features and target data
y = df['y']
X = df.drop(columns='y')

# 3. Encode categorical variables with `get_dummies`
# Encode the features dataset's categorical variables using get_dummies
X = pd.get_dummies(X)

# Review the features DataFrame
X.head()

# 4. Split the data into training and testing sets
# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# Review the distinct values from y
y_train.value_counts()


# 5. Scale the data using `StandardScaler`
# Instantiate a StandardScaler instance
scaler = StandardScaler()

# Fit the training data to the standard scaler
X_scaler = scaler.fit(X_train)

# Transform the training data using the scaler
X_train_scaled = X_scaler.transform(X_train)

# Transform the testing data using the scaler
X_test_scaled = X_scaler.transform(X_test)

## RandomForestClassifier
# 6. Create and fit a `RandomForestClassifier` to the **scaled** training data.
# Import the RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# Instantiate a RandomForestClassifier instance
model = RandomForestClassifier()

# Fit the training data to the model
model.fit(X_train_scaled, y_train)

### 7. Make predictions using the scaled testing data.
# Predict labels for original scaled testing features
y_pred = model.predict(X_test_scaled)


## Random Undersampler
# 8. Import `RandomUnderSampler` from `imblearn`.
# Import RandomUnderSampler from imblearn
from imblearn.under_sampling import RandomUnderSampler

# Instantiate a RandomUnderSampler instance
rus = RandomUnderSampler(random_state=1)

# 9. Fit the random undersampler to the scaled training data.
# Fit the training data to the random undersampler model
X_undersampled, y_undersampled = rus.fit_resample(X_train_scaled, y_train)


# 10. Check the `value_counts` for the undersampled target.
# Count distinct values for the resampled target data
y_undersampled.value_counts()


# 11. Create and fit a `RandomForestClassifier` to the **undersampled** training data.
# Instantiate a new RandomForestClassier model
model_undersampled = RandomForestClassifier()

# Fit the undersampled data the new model
model_undersampled.fit(X_undersampled, y_undersampled)

# 12. Make predictions using the scaled testing data.
# Predict labels for oversampled testing features
y_pred_undersampled = model_undersampled.predict(X_test_scaled)


# 13. Generate and compare classification reports for each model.
#  * Print a classification report for the model fitted to the original data
#  * Print a classification report for the model fitted to the undersampled data

# Print classification reports
print(f"Classifiction Report - Original Data")
print(classification_report(y_test, y_pred))
print("---------")
print(f"Classifiction Report - Undersampled Data")
print(classification_report(y_test, y_pred_undersampled))