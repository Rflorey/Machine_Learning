# Import required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 

'''
# Dataset:  occupancy.csv
Source: Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.
Description: Experimental data used for binary classification (room occupancy) from Temperature,Humidity,Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.
Variables/Columns
- Temperature, in Celsius
- Relative Humidity %
- Light in Lux
- CO2 in ppm
- Humidity Ratio, Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air
- Occupancy 0 or 1 
    - 0 for not occupied
    - 1 for occupied 
    '''
    
# Import data
# file_path = "https://static.bc-edx.com/mbc/ai/m4/datasets/occupancy.csv"
# df = pd.read_csv(file_path)
df = pd.read_csv('SVM/data/occupancy.csv')
print(df.head())


## Split the data into training and testing sets

# Get the target variable (the "Occupancy" column)
y = df["Occupancy"]


# Get the features (everything except the "Occupancy" column)
X = df.copy()
X = X.drop(columns="Occupancy")
print(X.head())


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


## Model and Fit to a Support Vector Machine
# Create the support vector machine classifier model with a 'linear' kernel
model = SVC(kernel='linear')

# Fit the model to the training data
model.fit(X_train, y_train)

# Validate the model by checking the model accuracy with model.score
print('Train Accuracy: %.3f' % model.score(X_train, y_train))
print('Test Accuracy: %.3f' % model.score(X_test, y_test))


## Predict the Testing Labels
# Make and save testing predictions with the saved SVM model using the testing data
testing_predections = model.predict(X_test)

# Review the predictions
print(testing_predections)

## Evaluate the Model
# Display the accuracy score for the testing dataset
accuracy_score(y_test, testing_predections)