#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Oct 12 11:15:47 2021

@author: Jamal(ilhamjamaluddin09@gmail.com)
"""

#Import python packages
import os
from osgeo import gdal, gdal_array
from pyrsgis import raster
from pyrsgis.convert import changeDimension
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
gdal.UseExceptions()
gdal.AllRegister()
import numpy as np
import pandas as pd
import cv2
from skimage.util import view_as_windows
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import normalize, to_categorical
from patchify import patchify, unpatchify
from numpy import save

#Define location of satellite image dataset
img1 = 'DATA/Pre_AOI1.tif' 
img2 = 'DATA/Post_AOI1.tif'
img3 = 'DATA/Pre_AOI2.tif'
img4 = 'DATA/Post_AOI2.tif'


#Define location of satellite visually interpreted label dataset
lbl1 = 'DATA/AOI1_GT.tif'
lbl2 = 'DATA/AOI2_GT.tif'


#Load satellite image and labelled image dataset in all AOI
#aoi1
ds1, pre1 = raster.read(img1, bands='all')
pre1 = np.moveaxis(pre1, 0, -1)
ds1, post1 = raster.read(img2, bands='all')
post1 = np.moveaxis(post1, 0, -1)
ds1, roi1 = raster.read(lbl1, bands=1)

#aoi2
ds2, pre2 = raster.read(img3, bands='all')
pre2 = np.moveaxis(pre2, 0, -1)
ds2, post2 = raster.read(img4, bands='all')
post2 = np.moveaxis(post2, 0, -1)
ds2, roi2 = raster.read(lbl2, bands=1)


print(pre1.shape, post1.shape, roi1.shape)
print(pre2.shape, post2.shape, roi2.shape)


#Make patches image (64x64)
#Cropped Pre_event
#AOI1
pre_event1 = patchify(pre1, (64, 64, 10), step=32).reshape(-1,64, 64, 10)
#AOI2
pre_event2 = patchify(pre2, (64, 64, 10), step=32).reshape(-1,64, 64, 10)

pre_event= np.concatenate((pre_event1,pre_event2))
print(pre_event.shape)


#Cropped Post_event
#AOI1
post_event1 = patchify(post1, (64, 64, 10), step=32).reshape(-1,64, 64, 10)
#AOI2
post_event2 = patchify(post2, (64, 64, 10), step=32).reshape(-1,64, 64, 10)

post_event = np.concatenate((post_event1,post_event2))
print(post_event.shape)


#Cropped label
label1 = patchify(roi1, (64, 64), step=32).reshape(-1,64, 64)
label2 = patchify(roi2, (64, 64), step=32).reshape(-1,64, 64)
roi = np.concatenate((label1,label2))
print(roi.shape)

print('Pre_Event:', pre_event.shape)
print('Post_Event:', post_event.shape)

#Make empty array for convlstm input data
#Default input for ConvLSTM using keras tensorflow is 5D tensor (samples, time, rows, cols, channels)
train = np.zeros(shape=(post_event.shape[0],2,post_event.shape[2],post_event.shape[2],post_event.shape[3]),dtype=np.float32)
#Add pre and post event data into empty array
train[:,0,:,:,0:post_event.shape[3]] = pre_event[:,:,:,0:post_event.shape[3]]
train[:,1,:,:,0:post_event.shape[3]] = post_event[:,:,:,0:post_event.shape[3]]
print(train.shape)

#Split all dataset to training, validation, and testing dataset
from sklearn.model_selection import train_test_split
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
train_data, test_data, train_label, test_label = train_test_split(train, roi, test_size=1 - train_ratio, random_state=1)
valid_data, test_data, valid_label, test_label = train_test_split(test_data, test_label, test_size=test_ratio/(test_ratio + validation_ratio), random_state=1) 

#Save numpy file
from numpy import save
save('NPY/train_data_70.npy', train_data)
save('NPY/train_label_70.npy', train_label)
save('NPY/valid_data_15.npy', valid_data)
save('NPY/valid_label_15.npy', valid_label)
save('NPY/test_data_15.npy', test_data)
save('NPY/test_label_15.npy', test_label)

print(train_data.shape, train_label.shape)
print(valid_data.shape, valid_label.shape)
print(test_data.shape, test_label.shape)



