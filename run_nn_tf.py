"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Enora Rice
Date: 11/20/20
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from cnn import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch


def load_data():
    #convert images to grayscale arrays
    hold_types = ["edges", "jugs", "pinches", "pockets", "slopers", "crimps"]
    cwd = os.getcwd()
    directory = cwd + "/scraped_data/"
    
    x_list = []
    y_list = []
    name_list = []

    for h in range(6):
        dir_string = directory + hold_types[h] + "/identified_holds"
        for f in os.listdir(dir_string) :
            if not f.startswith('.'):
                img_name = dir_string + "/" + f
                image = Image.open(img_name).convert("L") #(400, 400, 3)
                img_array = np.asarray(image)
                x_list.append(img_array)
                y_list.append(h)
                name_list.append(img_name)
                
    x_train = np.asarray(x_list)
    x_len, x_height, x_width = np.shape(x_train)
    x_train = x_train.reshape(x_len, x_height, x_width, 1)
    y_train = np.asarray(y_list)
    name = np.asarray(name_list)
    p = np.arange(len(y_list)) #shuffle with same seed
    np.random.shuffle(p)
    return (x_train[p], y_train[p])

def prepare_data(num_training=1068, num_validation=230, num_test=229):
    #Subsample the data and normalize
    (X_data, y_data) = load_data()

    X_data = np.asarray(X_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.int64).flatten()
    
    # Subsample the data
    mask = range(num_training)
    X_train = X_data[mask]
    y_train = y_data[mask]

    mask = range(num_training, num_training + num_validation)
    X_val = X_data[mask]
    y_val = y_data[mask]

    mask = range(num_training + num_validation, num_training + num_validation+ num_test)
    print("mask:")
    print(mask)
    X_test = X_data[mask]
    y_test = y_data[mask]

    # normalize the data. First find the mean and std of the *training* data,
    # then subtract off this mean from each dataset and divide by the std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel)/std_pixel
    X_test = (X_test - mean_pixel)/std_pixel
    X_val = (X_val - mean_pixel)/std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_step(images,labels, model, loss_object, optimizer): 
    # compute the predictions given the images, then compute the loss
    # compute the gradient with respect to the model parameters (weights), then
    # apply this gradient to update the weights (i.e. gradient descent)
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def val_step(images, labels, model, loss_object): 
    # compute the predictions given the images, then compute the loss
    predictions = model(images)
    loss = loss_object(labels, predictions)
    return loss, predictions

def run_training(model, train_dset, val_dset):
    plot_val = list()
    plot_train = list()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='val_accuracy')

    # train for 10 epochs (passes over the data)
    for epoch in range(6):
        for images, labels in train_dset:
            loss, predictions = train_step(images, labels, model, loss_object, optimizer)
            train_loss(loss)
            train_accuracy(labels, predictions)

    #loop over validation data and compute val_loss, val_accuracy too
        for images, labels in val_dset:
            loss, predictions = val_step(images, labels, model, loss_object)
            val_loss(loss)
            val_accuracy(labels, predictions)
       
        plot_val.append(val_accuracy.result())
        plot_train.append(train_accuracy.result())

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            val_loss.result(),
                            val_accuracy.result()*100))

    # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

    #Train vs. validation accuracy 
    # plt.plot(range(1,11), plot_val, 'bo-')
    # plt.plot(range(1,11), plot_train, 'ro-')
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.title("CNN model")
    # plt.legend(("Validation", "Training"))
    # plt.show()


def main():
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    print('Train data shape: ', X_train.shape)              # (49000, 32, 32, 3)
    print('Train labels shape: ', y_train.shape)            # (49000,)
    print('Validation data shape: ', X_val.shape)           # (1000, 32, 32, 3)
    print('Validation labels shape: ', y_val.shape)         # (1000,)
    print('Test data shape: ', X_test.shape)                # (10000, 32, 32, 3)
    print('Test labels shape: ', y_test.shape)              # (10000,)

    
    # set up train_dset, val_dset, and test_dset:
    train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10,000).batch(64)
    test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    val_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    

    # call the train function to train a three-layer CNN
    cnn_model = CNNmodel()
    run_training(cnn_model, train_dset, val_dset)
    model.save('saved_model/6_epochs')
    test(test_dset, cnn_model)
   

main()
