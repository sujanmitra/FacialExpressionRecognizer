from PIL import Image
import numpy as np
#from mnist import MNIST
import os
import struct

def euDCalc(i1, i2):
	return np.sum(np.sqrt((i1-i2)**2))
'''	
def toGray(img):
	print(img)
	n=np.empty((len(img),len(img[0]),3))
	for i in range(len(img)):
		for j in range(len(img[0])):
			gray=(img[i,j][0]*0.299)+(img[i,j][1]*0.587)+(img[i,j][2]*0.114)
			n[i, j]=(int(gray),int(gray),int(gray))
	return n
'''
def read(dataset = "training", path = "."):
	if dataset is "training":
		fname_img = os.path.join(path, 'train-images-idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
	elif dataset is "testing":
		fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
		raise ValueError("dataset must be 'testing' or 'training' ")
	with open(fname_lbl, 'rb') as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		lbl = np.fromfile(flbl, dtype=np.int8)
	with open(fname_img, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
	get_img = lambda idx: (lbl[idx], img[idx])
	for i in range(len(lbl)):
		yield get_img(i)

def toBinInv(img):
	#iarr=toGray(np.asarray(img))
	iarr=np.asarray(img)
	niarr=np.zeros((len(iarr),len(iarr[0])))
	for i in range(len(iarr)):
		for j in range(len(iarr[0])):
			 if iarr[i,j]!=0:
			 	niarr[i,j]=255
	img1=Image.fromarray(niarr)
	return img1

count=0
training_data=list(read(dataset = "training", path = "/home/sujanm/PythonProjects/Datasets"))

#lbl,img2=training_data[25]#Random Index Given
#Image.fromarray(img2).save('/home/sujanm/PythonProjects/Pictures/di.png')#Random Dataset Image Extracted
img1=Image.open('/home/sujanm/PythonProjects/Pictures/di.png')
i1arr=np.asarray(img1)
dumChk=np.zeros((len(i1arr),len(i1arr[0])))
img1a=np.zeros((28,28))
for i in range(len(i1arr)-27):
		for j in range(len(i1arr[0])-27):
			if dumChk[i,j]==0 :
				for k in range(28):
					for l in range(28):
						img1a[k,l]=i1arr[i+k,j+l][0]
				img4=toBinInv(Image.fromarray(img1a))
				#img4.show()
				for x in range(len(training_data)):
					lbl,img2=training_data[x]#Dataset Label,Image
					img3=toBinInv(img2)
					if euDCalc(np.asarray(img3),np.asarray(img4))==0:
						print("Match Found: ",lbl)
						for k in range(28):
							for l in range(28):
								dumChk[i+k,j+l]=-1
						count+=1
			dumChk[i,j]=-1
			print(i,j)
if count==0:
	print("No Match Found")
