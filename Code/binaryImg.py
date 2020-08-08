#Image Binarization
from PIL import Image
import numpy as np

def toGray(img):
	n=np.empty((len(img),len(img[0])),dtype='float32')
	for i in range(len(img)):
		for j in range(len(img[0])):
			gray=(img[i,j][0]*0.299)+(img[i,j][1]*0.587)+(img[i,j][2]*0.114)
			n[i, j]=(int(gray))
	return n

img=Image.open('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Pictures\\space.png')
iarr=toGray(np.asarray(img,dtype='float32'))
niarr=np.empty((len(iarr),len(iarr[0])),dtype='float32')
for i in range(len(iarr)):
	for j in range(len(iarr[0])):
		 if iarr[i,j]>165:
		 	niarr[i,j]=0
		 else:
		 	niarr[i,j]=255
img2=Image.fromarray(niarr)
img2.show()
#img2.convert('RGB').save('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Pictures\\dia.png')