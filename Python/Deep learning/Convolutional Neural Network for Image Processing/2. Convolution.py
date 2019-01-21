import os
os.chdir('E:\Datacamp\Python\Deep learning\Convolutional Neural Network for Image Processing')
# Import matplotlib
import matplotlib.pyplot as plt
im = plt.imread('bricks.png')[:,:,0]


#### CONVOLUTION
# Natural images contain spatial correlations. For example, pixels along a contour or edge
# Convoluion uses kernel with predefined patterns to search for the same pattern over the matrix


#### ONE DIMENSIONAL CONVOLUTION
# A convolution of an one-dimensional array with a kernel comprises of taking the kernel, sliding it along the array, multiplying it with the items in the array that overlap with the kernel in that location and summing this product.
import numpy as np
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)


#### IMAGE CONVOLUTION
# The convolution of an image with a kernel summarizes a part of the image as the sum of the multiplication of that part of the image with the kernel. 
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)
plt.imshow(im)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
print(result)
plt.imshow(result)


#### DEFINING IMAGE CONVOLUTION KERNELS
# Define a kernel that finds horizontal lines in images.
kernel = np.array([[-1, -1, -1], 
                   [1, 1, 1],
                   [-1, -1 ,-1]])
    
# Define a kernel that finds a light spot surrounded by dark pixels.
kernel = np.array([[-1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, -1]])
    
# Define a kernel that finds a dark spot surrounded by bright pixels.
kernel = np.array([[1, 1, 1],
                   [1, -1, 1],
                   [1, 1, 1]])
    
    
#### ADD CONVOLUTION TO NEURAL NETWORK = CONVOLUTIONAL NETWORK
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

# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Initialize the model object
model = Sequential()

# Add a convolutional layer
# kernel size = 3 x 3
# there are 10 kernel layers 3 x 3 which are created randomly (see 'Our CNN' - chap 2)
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(28, 28, 1))) # The train data has 50 images of size 28 x 28 with 1 color dimension

# Flatten the output of the convolutional layer
# flatten matrix size 28 x 28 to 784, for neural network
# Flatten takes 3 units out of 10 units of the output above
model.add(Flatten())

# Add an output layer for the 3 categories
model.add(Dense(3, activation='softmax'))

# Compile the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10) # 10 images per batch

# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)


#### PADDING IN CNN 
# See padding.gif
# Padding allows a convolutional layer to retain the resolution of the input into this layer. This is done by adding zeros around the edges of the input image, so that the convolution kernel can overlap with the pixels on the edge of the image.
# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(28, 28, 1), 
                 padding='same')) # output has the same size as the input
                                    # `"same"` is slightly inconsistent across backends with `strides` != 1, as described [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


#### STRIDES IN CNN
# See image padding_strides.gif
# The size of the strides of the convolution kernel determines whether the kernel will skip over some of the pixels as it slides along the image.
# This affects the size of the output because when strides are larger than one, the kernel will be centered on only some of the pixels.
# example, a matrix (1,2,3,4,5,.. n) x (a, b, c, .. z). first, kernel multiply with (123, abc), then (123, cde) but skip (123, bcd) 
# Construct a neural network with a Conv2D layer with strided convolutions that skips every other pixel.
# Initialize the model
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(28, 28, 1), 
                 strides=2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#### SIZE OF CONVOLUTIONAL LAYER OUTPUT(see slides)


#### DILATION IN CNN
# See images dilation.gif
# use dilation_rate


#### CNN WITH MULTIPLE CONVOLUTION LAYERS
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))
              
# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)
model.summary()


#### PARAMETERS IN DEEP CNN
# See slides
# Summarize the model 
model.summary()


#### REDUCE PARAMETERS WITH POOLINGS = REDUCE THE DIMENSION(1080P to 720P)
# replace 2 x 2 pixels with 1 pixels with the highest value
# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])

plt.imshow(result)


#### KERAS POOLING LAYER
from keras.layers import MaxPool2D

model = Sequential()
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary() 
# => model deeper but with less params

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)


#### PLOT LEARNING CURVES
import matplotlib.pyplot as plt

# Train the model and store the training object
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3, batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()


#### STORE OPTIMAL PARAMETERS
from keras.callbacks import ModelCheckpoint
# This checkpoint object will store the model parameters
# in the file "weights.hdf5"
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss',
save_best_only=True)

# Store in a list to be used during training
callbacks_list = [checkpoint]

# Fit the model on a training set, using the checkpoint as a callback
model.fit(train_data, train_labels, validation_split=0.2, epochs=3,
callbacks=callbacks_list)

# Load the weights from file
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data[:3])


#### REGULARIZATION: DROPOUT
# see dropout.png
# this technique works as ensemble learning when in each learning step:
    # select a subset of units, ignore forward pass and back-propagation
# => regularization because some parts may be sensitive with A, but other part sensitive with B => final model will sensitive with both A and B
from keras.layers import Dropout
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add a dropout layer
model.add(Dropout(0.2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


#### REGULARIZATION: BATCH NORMALIZATION
#  rescales the outputs of a layer to make sure that they have mean 0 and standard deviation 1
from keras.layers import BatchNormalization
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


#### DISHARMONY BETWEEN DROPOUT AND BATCH NORMALIZATION
# DROPOUT: slow down learning, make it more incremental and careful
# BATCH NORMALIZATION: faster learning
# => worse when perform together


#### EXTRACT KERNAL FROM TRAINED NETWORK
model = Sequential()
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Load the weights into the model
model.load_weights('weights.hdf5')

# See all the layers
model.layers

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][...,0, 0]
print(kernel)
plt.imshow(kernel)


#### VISUALIZE KERNEL RESPONSE
# Get the image number 3 from test data
test_image = test_data[7, :, :, 0]
plt.imshow(test_image)

filtered_image = np.zeros(test_image.shape)

# Output array
for ii in range(test_image.shape[0] - 2):
    for jj in range(test_image.shape[1] - 2):
        filtered_image[ii, jj] = (test_image[ii:ii+2, jj:jj+2] * kernel).sum()

# Print result
plt.imshow(filtered_image)


#### RESIDUAL NETWORK
#### FULLY CONVOLUTIONAL NETWORK
# https://medium.com/self-driving-cars/literature-review-fully-convolutional-networks-d0a11fe0a7aa