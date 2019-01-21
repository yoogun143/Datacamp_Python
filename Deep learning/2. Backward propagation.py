#### HOW WEIGHTS CHANGES AFFECT ACCURACY ON SINGLE PREDICTION (PIC 3)
# The data point you will make a prediction for
import numpy as np
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Define predict_with_network()
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)
    
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)
    
# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [-1, 2],
             'output': [-1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)


#### HOW WEIGHTS CHANGES AFFECT ACCURACY ON MANY PREDICTIONS = DATASET
# Use MSE as loss function
from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []
# Create actual value label
target_actuals = [1, 3, 5, 7]
# Create input_data
input_data_many = [np.array([0, 3]), np.array([1, 2]), np.array([-1, -2]), np.array([4, 0])]

# Loop over input_data
for row in input_data_many:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)


#### CALCULATING SLOPE
# This is the example calculating slope for the final layer only and the target layers, not relate to the input layer or first hidden layer.
# In other words, This network does not have any hidden layers, and it goes directly from the input (with 3 nodes) to an output node.
# Slope with MSE loss function is 2 * input_data * error
input_data = np.array([1, 2, 3])
weights = np.array([0, 2, 1])
target = 0

# Calculate the predictions: preds
preds = (input_data * weights).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)

# Set the learning rate: learning_rate
learning_rate = 0.01

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)


#### MULTIPLE UPDATES TO WEIGHTS
input_data = np.array([1, 2, 3])
target = 0
weights = np.array([0, 2, 1])
import matplotlib.pyplot as plt
n_updates = 20
mse_hist = []

# Define function
def get_slope(input_data, target, weights):
    # Calculate the predictions: preds
    preds = (input_data * weights).sum()

    # Calculate the error: error
    error = preds - target

    # Calculate the slope: slope
    slope = 2 * input_data * error

    # Print the slope
    return(slope)

def get_mse(input_data, target, weights):
    # Calculate the predictions: preds
    preds = (input_data * weights).sum()

    # Calculate the mse: mse
    mse = (preds - target)**2
    
    return(mse)
    
# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
# => MSE decreases as iteration go up


##### CALCULATING SLOPES OF ANY WEIGHT
#Gradients for weight is product of:
# 1. Node value feeding into that weight
# 2. Slope of activation function for the node being fed into
# 3. Slope of loss function with relative to output node


#### STOCHASTIC GRADIENT DESCENT
#It is common to calculate slopes on only a subset of the data (‘batch’)
#● Use a different batch of data to calculate the next update
#● Start over from the beginning once all data is used
#● Each time through the training data is called an epoch
#● When slopes are calculated on one batch at a time: stochastic gradient descent