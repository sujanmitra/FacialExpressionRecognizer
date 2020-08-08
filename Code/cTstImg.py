from PIL import Image
import numpy as np

a=np.asarray(Image.open('/home/sujanm/Pictures/di.png'))
b=np.zeros((200,100))
'''
for k in range(28):
	for l in range(28):
		b[0+k,0+l]=a[k,l]
'''
for k in range(28):
	for l in range(28):
		b[0+k,1+l]=a[k,l]
Image.fromarray(b).convert('RGB').save('/home/sujanm/PythonProjects/Pictures/di.png')
