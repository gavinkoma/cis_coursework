#fuck me
#please

nr_seed = 2019
import numpy as np
np.random.seed(nr_seed)
import tensorflow as tf

tf.set_random_seed = nr_seed

#%% import libraries
# import libraries
import json
import math
from tqdm import tqdm, tqdm_notebook
import gc
import warnings
import os

import cv2
from PIL import Image

import pandas as pd
import scipy
import matplotlib.pyplot as plt

from keras import backend as K
from keras import layers
from keras.applications.resnet import ResNet50
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

warnings.filterwarnings("ignore")

#%%

im_size = 224

BATCH_SIZE = 32

new_train = pd.read_csv('/Users/gavinkoma/Desktop/predictive/data/train.csv', sep = ',')
print(new_train.shape)

# path columns
new_train['id_code'] = '/Users/gavinkoma/Desktop/predictive/data/train/train_images/' + new_train['id_code'].astype(str) + '.png'

train_df = new_train.copy()
train_df.head()

# Let's shuffle the datasets
train_df = train_df.sample(frac=1).reset_index(drop=True)
print(train_df.shape)


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)
    
    return img

def preprocess_image_old(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)
    
    return img

# validation set
N = train_df.shape[0]
x_val = np.empty((N, im_size, im_size, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm_notebook(train_df['id_code'])):
    x_val[i, :, :, :] = preprocess_image(
        f'{image_id}',
        desired_size = im_size
    )

y_train = train_df['diagnosis'].values
y_train = train_df['diagnosis'].values

print(y_train.shape)
print(x_val.shape)
print(y_train.shape)

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype) 
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1): 
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])
    y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype) 
    y_train_multi[:, 4] = y_train[:, 4]

for i in range(3,-1,-1):
    y_train_multi[:,i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])
    
print("Y_train multi: {}".format(y_train_multi.shape)) 
print("Y_val multi: {}".format(y_train_multi.shape))




