
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
import keras
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv('./HAM10000_metadata.csv')
print(data.head())
print(data['dx'].value_counts())

path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('./', '*', '*.jpg'))}

img_path = pd.DataFrame(path.items(), columns=['id', 'path'])

###############################################################################
y=list(data['dx'].value_counts())
x=list(data['dx'].unique())

plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 15})
plt.bar(x,y)
plt.ylabel('Count')
plt.xlabel('Cell Type')
###############################################################################

le = LabelEncoder()
le.fit(data['dx'])
data['label'] = le.transform(data["dx"]) 
#print(data.sample(10))

print(data['dx'].value_counts())
#print(data['label'].value_counts())
label_list=['akiec','bcc','blk','df','mel','nv','vasc']


L0_path=[]
L1_path=[]
L3_path=[]
L6_path=[]
path = list(data.loc[data['label'] == 6, 'image_id'])
for j in range(len(path)):
    p = list(img_path.loc[img_path['id'] == path[j], 'path'])[0]
    img = np.asarray(Image.open(p).resize((384,256)))
    flip_lr = np.fliplr(img)
    flip_ud = np.flipud(img)
    L6_path.extend([img, flip_lr, flip_ud])

L2_path=[]
L4_path=[]
L5_path=[]
Path = list(data.loc[data['label'] == 5, 'image_id'])[:1000]
for j in range(len(Path)):
    p = list(img_path.loc[img_path['id'] == Path[j], 'path'])[0]
    img = np.asarray(Image.open(p).resize((384,256)))
    L5_path.append(img)
    


from sklearn.utils import resample
L0_balanced = resample(np.array(L0_path), replace=True, n_samples=500, random_state=42) 
L1_balanced = resample(np.array(L1_path), replace=True, n_samples=500, random_state=42) 
L2_balanced = resample(np.array(L2_path), replace=True, n_samples=500, random_state=42)
L3_balanced = resample(np.array(L3_path), replace=True, n_samples=500, random_state=42)
L4_balanced = resample(np.array(L4_path), replace=True, n_samples=500, random_state=42)
L5_balanced = resample(np.array(L5_path), replace=True, n_samples=500, random_state=42)
L6_balanced = resample(np.array(L6_path), replace=True, n_samples=500, random_state=42)


balanced_list=[L0_balanced,L1_balanced,L2_balanced,L3_balanced,L4_balanced,L5_balanced,L6_balanced]   

for i in range(len(label_list)):
    
    os.mkdir('./Data/' + label_list[i] + "/")
    for j in range(len(balanced_list[i])):
        cv2.imwrite('./Data/' + label_list[i] + '/' + 'Image_{}.jpg'.format(j), cv2.cvtColor(balanced_list[i][j],cv2.COLOR_RGB2BGR))
        
    
