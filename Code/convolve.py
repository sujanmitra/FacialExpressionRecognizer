import numpy as np

arr1=np.asarray([[0,1],[0,1]])
arr2=np.asarray([[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])
print(np.convolve(arr2,arr1))