import os
os.chdir('E:\Datacamp\Python\Deep learning')
import pandas as pd
df = pd.read_csv('hourly_wages.csv')

target = df.iloc[:, 0].values
predictors = df.iloc[:,1:].values.reshape(-1,9)

#### REGRESSION
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
model.fit(predictors, target, epochs = 20)


#### CLASSIFICATION
df = pd.read_csv('titanic_all_numeric.csv')
predictors = df.iloc[:,1:].values.reshape(-1,10)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax')) # softmax for classification

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, epochs = 20)

# Calculate predictions: predictions
pred_data = predictors[1:10,:]
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]


#### SAVE MODEL
model.save('test.h5')
from keras.models import load_model
my_model = load_model('test.h5')
my_model.summary()


#### CHANGE OPTIMIZATION PARAMETER: LEARNING RATE
# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [.000001, 0.01, 1]

# Define get new model: recreate model to avoid bias
def get_new_model():
    # Set up the model
    model = Sequential()

    # Add the first layer
    model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

    # Add the output layer
    model.add(Dense(2, activation='softmax'))
    return(model)
    
# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target, epochs = 10)
    

#### VALIDATION
# Commonly use validation split rather than cross vaidation
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, epochs = 10, validation_split=0.3)


#### EARLY STOPPING
# stop optimization when it isn't helping any more. => can set a high value for epochs
# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])


#### WORKFLOW FOR OPTIMIZING MODEL CAPACITY
# Start with small network
# gradually increase capacity with nodes per player and then hidden layers


#### MNIST DATASET
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = to_categorical(digits.target)

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(64,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs = 30, validation_split=0.3)
