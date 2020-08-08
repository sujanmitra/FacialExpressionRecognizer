#Compares two images calculating Euclidean Distance
import numpy as np

def euDCalc(i1, i2):
	return np.sum(np.sqrt((i1-i2)**2))

from PIL import Image
img1=Image.open('/home/sujanm/PythonProjects/Pictures/Img.jpg')
img2=Image.open('/home/sujanm/PythonProjects/Pictures/dImg.jpg')
i1=np.asarray(img1,dtype='float32')
i2=np.asarray(img2,dtype='float32')
if euDCalc(i1,i2)==0:
	print("Same")
else:
	print("Different")
