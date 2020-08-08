#Practice Space
'''
#Using OpenCV to compare two images
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim

def mse(img1,img2):
	e=np.sum((img1.astype(float)-img2.astype(float))**2)
	e/=(img1.shape[0]*img1.shape[1])
	return e

def compare(img1,img2):
	ss=ssim(img1,img2)
	me=mse(img1,img2)
	print("SSIM: ",ss)
	print("MSE: ",me)

img1=cv2.imread('/home/sujanm/PythonProjects/Pictures/Img.jpg')
img2=cv2.imread('/home/sujanm/PythonProjects/Pictures/dImg.jpg')
compare(img1,img2)
'''

'''
#Practicing Tensorflow
import tensorflow as tf
import cv2

a=tf.placeholder(float)
#init=tf.global_variables_initializer()
#tf.assign(a,5)
sess=tf.Session()
#sess.run(init)
img=cv2.imread('/home/sujanm/PythonProjects/Pictures/Img.jpg')
print(sess.run(a,feed_dict={a:img}))
sess.close()
'''

'''
#Learning to handle .csv file
import pandas as pd
import csv

#n=[str(1),'Tall Guy','CSE','May']
with open('tstData.csv','a') as f:#doubt
	w=csv.writer(f)
	w.writerow(n)
'''

'''
#Reading Image as Array without OpenCV
import numpy as np
from PIL import Image
img=Image.open('/home/sujanm/PythonProjects/Pictures/Img.jpg')
i=np.asarray(img,dtype='float32')
'''

'''
from PIL import Image
import numpy as np
img1=Image.open('/home/sujanm/PythonProjects/Pictures/di.png').convert('L')
i1arr=np.array(img1)
for i in range(len(i1arr)-27):
		for j in range(len(i1arr[0])-27):
			print(np.shape(i1arr[i:28,j:28].reshape(784,)))
			exit()
'''
'''
import filecmp
print(filecmp.cmp('file.txt', 'file2.txt'))
'''