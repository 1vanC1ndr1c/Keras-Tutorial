# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:18:30 2018

@author: Ivan
"""

###### Import libraries and modules ###########################################
import numpy as np
np.random.seed(123)  # for reproducibility


### Did not work without this####
from keras import backend as K  #
K.set_image_dim_ordering('th')  #
#################################



'''
Next, we'll import the Sequential model type from Keras. 
This is simply a linear stack of neural network layers, 
and it's perfect for the type of feed-forward CNN we're building 
in this tutorial.
'''
	
from keras.models import Sequential

'''
Next, let's import the "core" layers from Keras. 
These are the layers that are used in almost any neural network:
'''
	
from keras.layers import Dense, Dropout, Activation, Flatten

'''
Then, we'll import the CNN layers from Keras. 
These are the convolutional layers that will help us efficiently 
train on image data:
'''

from keras.layers import Convolution2D, MaxPooling2D

'''
Finally, we'll import some utilities.
This will help us transform our data later:
'''

from keras.utils import np_utils
###############################################################################









###### Load image data from MNIST #############################################
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()



print (X_train.shape)
# (60000, 28, 28)

from matplotlib import pyplot as plt
plt.imshow(X_train[0])
###############################################################################









###### Preprocess input data for Keras ########################################
'''
When using the Theano backend, you must explicitly declare a dimension for 
the depth of the input image. For example, 
a full-color image with all 3 RGB channels 
will have a depth of 3.

Our MNIST images only have a depth of 1, but we must explicitly declare that.

In other words, we want to transform our dataset from having shape 
(n, width, height) to (n, depth, width, height).
'''

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


print (X_train.shape)
# (60000, 1, 28, 28)

'''
The final preprocessing step for the input data is to convert our data type 
to float32 and normalize our data values to the range [0, 1].
'''

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
###############################################################################








###### Preprocess class labels for Keras ######################################
print (y_train.shape)
# (60000,)

'''
Hmm... that may be problematic. We should have 10 different classes, 
one for each digit, but it looks like we only have a 1-dimensional array. 
Let's take a look at the labels for the first 10 training samples:
'''

print (y_train[:10])
# [5 0 4 1 9 2 1 3 1 4]

'''
And there's the problem. The y_train and y_test data are not split into 
10 distinct class labels, but rather are represented as a single 
array with the class values.
'''

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print (Y_train.shape)
# (60000, 10)
###############################################################################








###### Define model architecture ##############################################
'''Let's start by declaring a sequential model format:'''
model = Sequential()

'''Next, we declare the input layer:'''
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))


'''The input shape parameter should be the shape of 1 sample. 
In this case, it's the same (1, 28, 28) that corresponds 
to  the (depth, width, height) of each digit image.

But what do the first 3 parameters represent? 
They correspond to the number of convolution filters to use, 
the number of rows in each convolution kernel, 
and the number of columns in each convolution kernel, respectively.
'''

print (model.output_shape)
# (None, 32, 26, 26)

'''
Next, we can simply add more layers to our model like we're building legos:
'''

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


'''
Again, we won't go into the theory too much, 
but it's important to highlight the Dropout layer we just added. 
This is a method for regularizing our model in order to prevent overfitting.
'''

'''
MaxPooling2D is a way to reduce the number of parameters in our model 
by sliding a 2x2 pooling filter across the previous 
layer and taking the max of the 4 values in the 2x2 filter.
'''


'''
So far, for model parameters,
we've added two Convolution layers. 
To complete our model architecture, 
let's add a fully connected layer and then the output layer:
'''

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

'''
For Dense layers, the first parameter is the output size of the layer. 
Keras automatically handles the connections between layers.

Note that the final layer has an output size of 10, 
corresponding to the 10 classes of digits.

Also note that the weights from the Convolution 
layers must be flattened (made 1-dimensional) 
before passing them to the fully connected Dense layer.
'''
###############################################################################







######### Compile model #######################################################
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
###############################################################################



###### Fit model on training data #############################################
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

# Epoch 1/10
# 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164
###############################################################################





########## Evaluate model on test data ########################################
score = model.evaluate(X_test, Y_test, verbose=0)
###############################################################################


