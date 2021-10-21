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
import tensorflow

##################### Load Data Part #############################
#Load NPY data
train_data = np.load("NPY/train_data_70.npy")
lbl_train = np.load("NPY/train_label_70.npy")
valid_data = np.load("NPY/valid_data_15.npy")
lbl_valid =np.load("NPY/valid_label_15.npy")
test_data = np.load("NPY/test_data_15.npy")
lbl_test = np.load("NPY/test_label_15.npy")

print(train_data.shape, lbl_train.shape)
print(valid_data.shape, lbl_valid.shape)
print(test_data.shape, lbl_test.shape)
np.unique(lbl_train)

#Change label data into categorical label data
train_label = tensorflow.keras.utils.to_categorical(lbl_train)
valid_label = tensorflow.keras.utils.to_categorical(lbl_valid)
test_label = tensorflow.keras.utils.to_categorical(lbl_test)
print(train_label.shape, valid_label.shape, test_label.shape)

#Display a image and label data to make sure all dataset have been loaded properly
img = train_data[64, 1,:, :, 5]
mask = lbl_train[64,:,:]
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.title('Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Mask')
plt.imshow(mask)
plt.show()

##################### Deep Learning Architecture Part #############################
#Constructing ConvLSTM + Semantic segmentation deep learning architecture
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense,ConvLSTM2D,Conv2DTranspose
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
import segmentation_models as sm

SM_FRAMEWORK=tf.keras
sm.set_framework('tf.keras') #set SM modul packages that using tf.keras

input_shape_ConvLSTM = 2, 64, 64, 10

#define ConvLSTM architecture
def ConvLSTM(input_shape_ConvLSTM):
    
    input = Input (input_shape_ConvLSTM)
    Convlstm = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same')(input)
    model = Model(input,Convlstm)
    
    return model

ConvLSTM = ConvLSTM(input_shape_ConvLSTM) #constructed convlstm model
ConvLSTM.summary()

#Define Unet architecture
UNET = sm.Unet('vgg16', classes=3, input_shape=(64, 64, 32), encoder_weights=None)
UNET.summary()

#Combine convlstm + Unet architecture
x=ConvLSTM.output #get output layer from convlstm model
x=UNET(x) #feed output layer from convlstm into unet model

Model=Model(inputs=ConvLSTM.input, outputs=x) #construct whole Model
Model.summary()


#Define hyperparamters for training model (training implementation details)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
import segmentation_models as sm

#Define batchsize
batch_sz=64

#Define loss and metrics function
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

#Define optimizer and learning rate
adam = optimizers.Adam(learning_rate=0.001)

#Define early stoppinf function (early stopping and save model weight based on validation loss)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30) 
mc = ModelCheckpoint('Model/10bands_lstm_UNET_VGG16_coba.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
callbacks_list = [mc,es]

#Compile Model with hyperparameters
Model.compile(loss=total_loss, optimizer=adam, metrics=metrics)

#Train the model
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
history = Model.fit(train_data, train_label,
              epochs=1000,
              batch_size=batch_sz,
              validation_data=(valid_data, valid_label),verbose=2,
              callbacks=callbacks_list)

#Load trained model weight
Model.load_weights('Model/10bands_lstm_UNET_VGG16_coba.h5')


#Plot training history
def plot_learning_curves(history,epoch,min_val,max_val):
        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.axis([0, epoch, min_val, max_val])
        plt.show()
plot_learning_curves(history,316,0,1)


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['iou_score'], label='iou_score')
plt.grid(True)
plt.axis([0, 261, 0, 1])
plt.ylabel('Value')
plt.xlabel('No. epochs')
plt.legend(loc="best")
plt.show()


plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['val_iou_score'], label='val_iou_score')
plt.grid(True)
plt.ylabel('Value')
plt.axis([0, 261, 0, 1])
plt.xlabel('No. epochs')
plt.legend(loc="best")
plt.show()


plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['val_f1-score'], label='val_f1-score')
plt.grid(True)
plt.ylabel('Value')
plt.axis([0, 261, 0, 1])
plt.xlabel('No. epochs')
plt.legend(loc="best")
plt.show()

##################### Accuracy Metrics Part #############################
#Intersection over Union (IoU)
from tensorflow.keras.metrics import MeanIoU

y_pred=Model.predict(test_data) #predict testing data using trained weight
y_pred_argmax=np.argmax(y_pred, axis=3)

n_classes = 3 #define number of classes
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(lbl_test[:,:,:], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy()) #mean IoU


#Calcultae IoU per each class
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)

#Calculate confusion metrics and accuracy metrics
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
ypred = y_pred_argmax.reshape(y_pred_argmax.shape[0]*y_pred_argmax.shape[1]*y_pred_argmax.shape[2])
ytest = lbl_test.reshape(lbl_test.shape[0]*lbl_test.shape[1]*lbl_test.shape[2])

target_names = ['Non-Mangrove', 'Mangrove', 'Mangrove-Deforestation']
classification = classification_report(ytest, ypred, target_names=target_names,digits=5)

print("{}".format(classification))
print ("Accuracy = ", accuracy_score(ytest, ypred))
print (confusion_matrix(ytest, ypred))

##################### Visualize testing result in each patches #############################
import random
test_img_number = random.randint(0, len(valid_data))
#test_img_number = 900
test_img = test_data[test_img_number,:,:,:,0:10]
ground_truth = lbl_test[test_img_number,:,:]

test_img = test_img.reshape(1,test_img.shape[0],test_img.shape[1],test_img.shape[2],test_img.shape[3])

print(test_img.shape,ground_truth.shape)

prediction = (Model.predict(test_img))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
print(prediction.shape)
print(predicted_img.shape)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(valid_data[test_img_number, 1,:, :, 5], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()




