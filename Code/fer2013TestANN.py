import os
import PIL
import cv2
from PIL import Image
import numpy as np

#data_path_test = 'C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\kaggle dataset\\fer2013\\PrivateTest\\'
data_path_test = 'C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\kaggle dataset\\fer2013\\PublicTest\\'
data_dir_test_list = os.listdir(data_path_test)

img_data_list = []

#Try to remove OpenCV
for dataset in data_dir_test_list:
    img_list = os.listdir(data_path_test+'/'+ dataset)
    #print('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path_test + '/' + dataset + '/' + img)
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img,(48, 48))
        img_data_list.append(input_img_resize)

img_data = np.asarray(img_data_list, dtype=np.float32)
img_data = img_data/255

from sklearn.utils import shuffle
import keras
from keras import backend as tf
from keras.utils import np_utils
from keras.models import load_model

num_classes = 7
num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,), dtype='int64')
#PrivateTest
'''
labels[0:490] = 0 
labels[491:545] = 1 
labels[546:1073] = 2 
labels[1074:1952] = 3 
labels[1953:2578] = 4 
labels[2579:3172] = 5 
labels[3173:] = 6 
'''
#PublicTest
labels[0:466] = 0 
labels[467:522] = 1 
labels[523:1018] = 2 
labels[1019:1913] = 3 
labels[1914:2520] = 4 
labels[2521:3173] = 5 
labels[3174:] = 6 

Y = np_utils.to_categorical(labels, num_classes)
X_test, y_test = shuffle(img_data, Y, random_state=2)#Normalization

model = load_model('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\SavedModels\\fer2013Mine.h5')
score = model.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', score[1])