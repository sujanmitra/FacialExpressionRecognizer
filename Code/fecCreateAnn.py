import os
import PIL
import cv2
from PIL import Image
import numpy as np

data_path = 'C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\training_data'
data_dir_list = os.listdir(data_path)

img_data_list = []

#Try to remove OpenCV
for dataset in data_dir_list:
    img_list = os.listdir(data_path+'/'+ dataset)
    #print('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img,(128, 128))
        img_data_list.append(input_img_resize)

img_data = np.asarray(img_data_list, dtype=np.float32)
img_data = img_data/255

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras import backend as tf
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import adam
from keras.regularizers import l2

num_classes = 7
num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,), dtype='int64')
labels[0:164] = 0 
labels[165:218] = 1 
labels[219:424] = 2 
labels[425:531] = 3 
labels[532:769] = 4 
labels[770:884] = 5 
labels[885:] = 6 

Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data, Y, random_state=2)#Normalization
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

input_shape = img_data[0].shape
num_features = 32

model = Sequential()

model.add(Conv2D(32, 5, 5, input_shape = input_shape, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, batch_size = 7, nb_epoch = 20, verbose = 1, validation_data = (X_test, y_test))
model.save(os.path.join('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\SavedModels', 'fecMine.h5'))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', score[1])