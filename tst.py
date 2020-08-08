import os
import PIL
import cv2
from PIL import Image
import numpy as np

data_path = 'C:\\Users\\S MITRA\\Downloads\\DRIVE\\test\\1st_manual'
data_dir_list = os.listdir(data_path)

img_data_list = []

img_list = os.listdir(data_path)
for img in img_list:
    input_img = cv2.imread(data_path + '/' + img)
    img_data_list.append(input_img)

img_data = np.asarray(img_data_list, dtype=np.float32)
img_data = img_data/255

import keras
from keras import backend as tf
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import adam
from keras.regularizers import l2

input_shape = img_data[0].shape
num_of_samples = img_data.shape[0]

model = Sequential()

model.add(Conv2D(32, 3, 3, input_shape = input_shape))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Conv2D(1, 3, 3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
model.fit(img_data, batch_size = 5, nb_epoch = 10, verbose = 1)
model.save(os.path.join('C:\\Users\\S MITRA\\Downloads\\DRIVE', 'retTst.h5'))