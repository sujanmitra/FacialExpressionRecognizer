import numpy as np
from PIL import Image
#from mlxtend.data import loadlocal_mnist
import keras
from keras import backend as tf
from keras.models import load_model

model=load_model('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\SavedModels\\alphanumEMNIST.h5')

img1=Image.open('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Pictures\\MyFolder\\di1.png').convert('L')
i1arr=np.array(img1)

for i in range(0,len(i1arr)-27,28):
		for j in range(0,len(i1arr[0])-27,28):
			predicted=model.predict(np.asmatrix(i1arr[i:(28+i),j:(28+j)].reshape(784)))
			print(predicted.argmax())
