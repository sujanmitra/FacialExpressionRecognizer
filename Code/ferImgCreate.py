import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv('C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\kaggle dataset\\fer2013\\fer2013.csv')
np_df = df[['pixels']].as_matrix()
np_df_em = df[['emotion']].as_matrix()
np_df_type = df[['Usage']].as_matrix()

#print(np.unique(np_df_type))

dir = 'C:\\Users\\S MITRA\\Documents\\Sujan\\PythonProjects\\Facial Expression Recognizer\\Datasets\\kaggle dataset\\fer2013\\Images\\'
for i in range(0, df.shape[0]):
    img = np_df[i][0].split(' ')
    img_out = np.zeros([48, 48])
    #Fuck Logic but Works
    count = 0
    for c1 in range(0, 48):
        for c2 in range(0, 48):
            img_out[c1, c2] = img[count]
            count+=1
    Image.fromarray(img_out).convert('RGB').save(dir+str(np_df_type[i][0])+'\\'+str(np_df_em[i][0])+'\\'+str(i)+'.png')
