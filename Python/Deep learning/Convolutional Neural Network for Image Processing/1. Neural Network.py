import os
os.chdir('E:\Datacamp\Python\Deep learning\Convolutional Neural Network for Image Processing')


#### IMAGE AS DATA: CHANGING IMAGE
# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()

# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()


#### ONE-HOT ENCODING TO REPRESENT IMAGES
# = to_categorical in keras
# The number of image categories
n_categories = 3

# The unique values of categories in the data
import numpy as np
categories = np.array(["shirt", "dress", "shoe"])

# Original labels to be encoded to matrix
labels = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1
    
    
#### EVALUATE A CLASSIFIER (ACCURACY)
test_labels = np.array([[0., 0., 1.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 1., 0.]])
    
predictions = np.array([[0., 0., 1.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 0.], # => wrong prediction
                        [0., 0., 1.],
                        [1., 0., 0.], # => wrong prediction
                        [0., 0., 1.],
                        [0., 1., 0.]])

# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum()
print(number_correct)

# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(predictions)
print(proportion_correct)


#### NEURAL NETWORK 
fashion = np.load('fashion.npz')
fashion.zip
lst = fashion.files
for item in lst:
    print(item)
    print(fashion[item])
matrix = fashion['arr_0']
list = matrix.tolist()
test_data = list['test_data']
test_labels = list['test_labels']
train_data = list['train_data']
train_labels = list['train_labels']

# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
# Dense layers, meaning that each unit in each layer is connected to all of the units in the previous layer.
model.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model.add(Dense(10, activation='relu'))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Reshape the data to two-dimensional array (50 images of flatten arrays)
train_data = train_data.reshape(50, 784)

# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

# Reshape test data (should use model.predict to get the test data)
test_data = test_data.reshape(10, 784)

# Evaluate the model
model.evaluate(test_data, test_labels)
