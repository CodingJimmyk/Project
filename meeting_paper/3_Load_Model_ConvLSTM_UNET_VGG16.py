#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Oct 12 11:15:47 2021

@author: Jamal (ilhamjamaluddin09@gmail.com)
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

#Load data
img1 = 'DATA/Test_Pre.tif'
img2 = 'DATA/Test_Post.tif'

ds1, pre = raster.read(img1, bands='all')
pre = np.moveaxis(pre, 0, -1)
ds2, post = raster.read(img2, bands='all')
post = np.moveaxis(post, 0, -1)
print(pre.shape)
print(post.shape)

#Construct crop patches images
H = post.shape[0]
W = post.shape[1]

y=0
x=0
h=64*(np.int(post.shape[0]/64))
w=64*(np.int(post.shape[1]/64))

crop_pre = pre[y:y+h, x:x+w]
crop_post = post[y:y+h, x:x+w]

#Cropped Pre_event
pre_event = patchify(crop_pre, (64, 64, post.shape[2]), step=64).reshape(-1,64,64,post.shape[2])
print(pre_event.shape)

#Cropped Post_event
post_event = patchify(crop_post, (64, 64, post.shape[2]), step=64).reshape(-1,64, 64, post.shape[2])
print(post_event.shape)

#Constract input data fro ConvLSTM
train = np.zeros(shape=(post_event.shape[0],2,post_event.shape[2],post_event.shape[2],post_event.shape[3]),dtype=np.float32)
train[:,0,:,:,0:post_event.shape[3]] = pre_event[:,:,:,0:post_event.shape[3]]
train[:,1,:,:,0:post_event.shape[3]] = post_event[:,:,:,0:post_event.shape[3]]
print(train.shape)

## Model_define
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense,ConvLSTM2D,Conv2DTranspose
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
import segmentation_models as sm

SM_FRAMEWORK=tf.keras
sm.set_framework('tf.keras')

def ConvLSTM(input_shape_ConvLSTM):
    
    input = Input (input_shape_ConvLSTM)
    Convlstm = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same')(input)
    model = Model(input,Convlstm)
    
    return model

input_shape_ConvLSTM = 2, 64, 64, 10

ConvLSTM = ConvLSTM(input_shape_ConvLSTM)
ConvLSTM.summary()

UNET = sm.Unet('vgg16', classes=3, input_shape=(64, 64, 32), encoder_weights=None)
UNET.summary()

x=ConvLSTM.output
x=UNET(x)
Model=Model(inputs=ConvLSTM.input, outputs=x)
Model.summary()

#Load trained model weight
Model.load_weights('Model/10bands_lstm_UNET_VGG16_coba.h5')

#Predict image
predict_image = Model.predict(train)
print(predict_image.shape)

prediction = tf.argmax(predict_image, axis=-1).numpy() 
print(prediction.shape)

#Test visualize result
plt.figure(figsize=(12, 8))
plt.subplot(232)
plt.title('Predicted')
plt.imshow(prediction[10,:,:], cmap='jet')

#Reconstract whole image
back = patchify(crop_post, (64, 64, post.shape[2]), step=64)
patches = prediction.reshape(back.shape[0],back.shape[1],64, 64)
print(patches.shape)

reconstructed_image = unpatchify(patches, (h, w))

#Export result with geospatial information
outFile = 'Predicted_result.tif'
raster.export(reconstructed_image, ds2, filename=outFile, dtype='float')




