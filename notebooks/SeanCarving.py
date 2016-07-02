
# coding: utf-8

# In[1]:

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

import scipy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

from skimage import transform
from skimage import color

# In[2]:

import os

dir_conts = os.popen("ls ../data/images/roof_images").read().split("\n")[:-1]
print len(dir_conts)


# In[3]:

all_ids = map(lambda x: x.strip(".jpg"), dir_conts)


# In[4]:

import pandas as pd

df_all = pd.DataFrame({"k" : all_ids})


# In[5]:

# id_train.csv  sample_submission4.csv
df_train = pd.read_csv("../data/csv_lables/id_train.csv")
df_test = pd.read_csv("../data/csv_lables/sample_submission4.csv")


# In[6]:

df_all.keys()


# In[7]:

unsupervised = set(map(int,list(df_all["k"]))) - set(list(df_train["Id"])) -set(list(df_test["Id"]))


# In[8]:

import cv2
print len(unsupervised)

unsupervised_list = list()
for img_id in unsupervised:
    img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
    resized = transform.resize(img, (64, 64) )
    unsupervised_list.append(resized)

unsupervised_matrix = np.array(unsupervised_list)
print unsupervised_matrix.shape

np.savez("../data/pkl/unsupervisedX.npz",x=unsupervised_matrix)

# In[9]:

# In[10]:

train = list()
trainY = list()
for img_id, img_label in zip(list(df_train["Id"]), list(df_train["label"])):
    img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
    resized = transform.resize(img, (64, 64) )

    if img_label == 1:
        print 'lol'
	num_rows, num_cols = resized.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
        train.append(img_rotation)
        trainY.append(2)
    elif img_label == 2:
	num_rows, num_cols = resized.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
        train.append(img_rotation)
        trainY.append(1)
    elif img_label == 3:
	num_rows, num_cols = resized.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
        train.append(img_rotation)
        trainY.append(img_label)

    train.append(resized)
    trainY.append(img_label)

    flipped = cv2.flip(resized,0)
    train.append(flipped)
    trainY.append(img_label)

    num_rows, num_cols = resized.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
    img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
    img_rotation = cv2.warpAffine(resized, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

    zoomed = cv2.resize(resized,None,fx=0.9, fy=0.9, interpolation = cv2.INTER_CUBIC)
    zm1 = np.zeros_like(resized)
    x = int(resized.shape[1]/2 - float(zoomed.shape[1]/2))
    y = int(resized.shape[1]/2 - float(zoomed.shape[0]/2))
    x_max = int(resized.shape[1]/2 + float(zoomed.shape[1]/2))
    y_max = int(resized.shape[1]/2 + float(zoomed.shape[0]/2))
    zm1[y:y_max, x:x_max] = zoomed
    train.append(zm1)
    trainY.append(img_label)

    num_rows, num_cols = flipped.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
    img_rotation = cv2.warpAffine(flipped, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
    img_rotation = cv2.warpAffine(flipped, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

    num_rows, num_cols = zm1.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
    img_rotation = cv2.warpAffine(zm1, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -5, 1)
    img_rotation = cv2.warpAffine(zm1, rotation_matrix, (num_cols, num_rows))
    train.append(img_rotation)
    trainY.append(img_label)

# In[11]:

train_matrix = np.array(train)
print train_matrix.shape

trainY_matrix = np.array(trainY)
print trainY_matrix.shape

#trainY = df_train["label"].values
#print trainY.shape

# In[12]:

test = list()
for i, img_id in enumerate(list(df_test["Id"])):
    img = cv2.imread("../data/images/roof_images/" + str(img_id) + ".jpg")
    resized = transform.resize(img, (64, 64) )
    test.append(resized)


# In[13]:

test_matrix = np.array(test)
print test_matrix.shape


# In[14]:


# In[19]:

np.savez_compressed("../data/pkl/trainX.npz",x=train_matrix, y=trainY_matrix)


# In[20]:

np.savez_compressed("../data/pkl/testX.npz", x=test_matrix)


# In[16]:

#trainY = df_train["label"].values


# In[17]:

#trainY


# In[18]:

#with open("../data/pkl/trainY.pkl","w") as f:
   # pkl.dump(trainY,f)


# In[ ]:


