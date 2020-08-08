import numpy as np
import cv2
 
import keras
from keras import backend as tf
from keras.models import load_model

names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
def getLabel(id):
    return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]

model = load_model('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\SavedModels\\fer2013Mine.h5')

img = np.array(cv2.imread('C:\\Users\\S MITRA\\Downloads\\shitImg.jpg'))#.jpg only

face_cascade = cv2.CascadeClassifier('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\haarcascade_frontalface_default.xml')

#For still images
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, a, a) in faces:
    crop_img = img[y:y+a, x:x+a]
    resized_img = cv2.resize(crop_img, (48, 48))
    test_image = np.array(resized_img).reshape(1, 48, 48, 3)
    cv2.rectangle(img, (x, y), (x+a, y+a), (255, 255, 255), 2)
    cv2.putText(img, getLabel(model.predict_classes(test_image)[0]), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('output', img)
cv2.waitKey()
'''
#For video feed

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, a, a) in faces:
        crop_img = img[y:y+a, x:x+a]
        resized_img = cv2.resize(crop_img, (48, 48))
        test_image = np.array(resized_img).reshape(1, 48, 48, 3)
        cv2.rectangle(img, (x, y), (x+a, y+a), (255, 255, 255), 2)
        cv2.putText(img, getLabel(model.predict_classes(test_image)[0]), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('output', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
