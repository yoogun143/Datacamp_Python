import numpy as np
import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Supervised learning with sklearn')
col_name = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration',
       'synfuels', 'education', 'superfund', 'crime', 'duty_free_exports',
       'eaa_rsa']
df = pd.read_csv('house-votes-84.data.txt', names = col_name)


#### DROPPING MISSING DATA
# Convert '?' to NaN
df[df == '?'] = np.NaN

# Other way to convert '?' to NaN
df = pd.read_csv('house-votes-84.data.txt', names = col_name)
df.replace('?', np.nan, inplace = True)

# Replace character with numeric
df[df == 'y'] = 1
df[df == 'n'] = 0

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


#### IMPUTE MISSING DATA IN A PIPELINE
df = pd.read_csv('house-votes-84.data.txt', names = col_name)
df[df == '?'] = np.NaN
df[df == 'y'] = 1
df[df == 'n'] = 0
y = df.party
X = df.drop('party', axis = 1)

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
