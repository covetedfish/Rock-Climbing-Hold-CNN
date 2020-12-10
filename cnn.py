"""
Convolutional neural network architecture.
Author: Enora Rice
Date: 11/20/20
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

##################

class CNNmodel(Model):
    """
    A convolutional neural network; the architecture is:
    Conv -> ReLU -> Conv -> ReLU -> Dense
    """
    def __init__(self):
        super(CNNmodel, self).__init__()
        # TODO complete constructor
        self.c1 = Conv2D(64, 5, activation = tf.nn.relu)
        self.pool = MaxPooling2D(pool_size= (8,8), strides = 2)
        self.c2 = Conv2D(32, 3, activation = tf.nn.relu)
        self.flatten = Flatten()
        self.dense = Dense(6, activation = tf.nn.softmax)

        # First conv layer: 32 filters, each 5x5
        # Second conv layer: 16 filters, each 3x3

    def call(self, x):
        c1 = self.c1(x)
        pool1 = self.pool(c1)
        c2 = self.c2(pool1)
        pool2 = self.pool(c2)
        flatten = self.flatten(c2)
        dense = self.dense(flatten)
        return dense

def three_layer_convnet_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    cnn_model = CNNmodel()

    # try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    x_np = np.zeros((64, 32, 32, 3))
    #x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = cnn_model.call(x_np)
    print(output) 


    for v in cnn_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test three layer function
    three_layer_convnet_test()

if __name__ == "__main__":
    main()
