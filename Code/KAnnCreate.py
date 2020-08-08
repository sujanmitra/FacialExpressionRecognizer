import os
import struct
import numpy as np
from PIL import Image
from mlxtend.data import loadlocal_mnist

x_train, y_train = loadlocal_mnist(
        images_path='C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\gzip\\emnist-byclass-train-images-idx3-ubyte', 
        labels_path='C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\gzip\\emnist-byclass-train-labels-idx1-ubyte')
x_test, y_test = loadlocal_mnist(
        images_path='C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\gzip\\emnist-byclass-test-images-idx3-ubyte', 
        labels_path='C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Datasets\\gzip\\emnist-byclass-test-labels-idx1-ubyte')

import tensorflow as tf
import keras
from keras.models import Sequential,load_model
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import *
from keras.layers import Activation, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
num_features=32

'''
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(62,activation=tf.nn.softmax))   
'''

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(27, activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=2)
model.save(os.path.join('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjectsMNIST\\SavedModels', 'alphanumEMNIST.h5'))
model.evaluate(x_test, y_test)
