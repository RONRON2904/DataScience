import numpy as np
import sys
sys.path.append("data_preprocessing/")
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
keras.__version__
 
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Dropout
from keras import regularizers
from keras import backend as K
 
import h5py
 
from keras.layers.normalization import BatchNormalization
 
from keras.optimizers import Nadam
from keras.callbacks import History
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from keras.backend import binary_crossentropy
 
import datetime
import os
 
import random
 
from keras.models import model_from_json
import threading
import itertools
from data import crop_impact_area, crop_crop, get_imp_nparr
from predict import get_ind
import matplotlib.pyplot as plt
 
img_rows = 128
img_cols = 128
num_channels = 1
num_mask_channels = 1
 
def get_unet0():
    if K.image_data_format() == 'channels_first':
        inputs = Input(shape=(num_channels, img_rows, img_cols))
    else:
        inputs = Input(shape=(img_rows, img_cols, num_channels))
    dropout_value = 0.2
    reg_val = 0.01
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Dropout(dropout_value)(conv1)
    conv1 = Conv2D(32, 3,  padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), activation='elu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
     
    conv2 = Conv2D(64, 3, padding='same',  kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Dropout(dropout_value)(conv2)
    conv2 = Conv2D(64, 3,  padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
     
    conv3 = Conv2D(128, 3, padding='same',  kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Dropout(dropout_value)(conv3)
    conv3 = Conv2D(128, 3,  padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
     
    conv4 = Conv2D(256, 3, padding='same',  kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Dropout(dropout_value)(conv4)
    conv4 = Conv2D(128, 3,  padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
     
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Dropout(dropout_value)(conv5)
    conv5 = Conv2D(512, 3,  padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv2 = UpSampling2D(size=(2, 2))(conv2)
    pool1 = UpSampling2D(size=(2, 2))(pool1)
    conv3 = UpSampling2D(size=(2, 2))(conv3)
    pool2 = UpSampling2D(size=(2, 2))(pool2)
    conv4 = UpSampling2D(size=(2, 2))(conv4)
    pool3 = UpSampling2D(size=(2, 2))(pool3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    pool4 = UpSampling2D(size=(2, 2))(pool4)
     
    up6 = keras.layers.concatenate([conv5, pool4], axis=3)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(up6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Dropout(dropout_value)(conv6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv6)
    conv6 = BatchNormalization(axis=1)(conv5)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
     
    conv6 = UpSampling2D(size=(2, 2))(conv6)
     
    up7 = keras.layers.concatenate([conv6, pool3], axis=3)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(up7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Dropout(dropout_value)(conv7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
     
    conv7 = UpSampling2D(size=(2, 2))(conv7)
     
    up8 = keras.layers.concatenate([conv7, pool2], axis=3)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(up8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Dropout(dropout_value)(conv8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(reg_val))(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
     
    conv8 = UpSampling2D(size=(2, 2))(conv8)
     
    up9 = keras.layers.concatenate([conv8, pool1], axis=3)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(reg_val))(up8)
    conv9 = BatchNormalization(axis=1)(conv8)
    conv9 = keras.layers.advanced_activations.ELU()(conv8)
    conv9 = Dropout(dropout_value)(conv9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val))(conv8)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
     
    conv9 = Conv2D(num_mask_channels, 1, activation='sigmoid')(conv9)
     
    model = Model(inputs=inputs, outputs=conv9)
    return model
 
if __name__ == '__main__':
    #    images = np.load('x_train.npy')
    #    masks = np.load('y_train.npy')
    images = np.load('images.npy')
    masks = np.load('masks.npy')
    fp_images = np.load('fp_img.npy')
    fp_masks = np.load('fp_msk.npy')
    #    fp_inds = np.load('false_positive_inds.npy')
    #    x_train = images[:10000]
    #y_train = masks[:10000]
    X = np.concatenate((images, fp_images))
    Y = np.concatenate((masks, fp_masks))
    fp_images = X[6020:]
    fp_masks = Y[6020:]
    x_train = np.concatenate((images[:3100], fp_images))
    y_train = np.concatenate((masks[:3100], fp_masks))
    #x_train, y_train = get_imp_nparr(x_train, y_train)
    #x_train1, y_train1 = get_imp_nparr(x_train, y_train)
    #x_train2, y_train2 = get_fp_arrays(x_train, y_train, fp_inds)
    #x_train = np.concatenate((x_train1, x_train2))
    #y_train = np.concatenate((y_train1, y_train2))   
    #x_train = x_train.astype('float16')
    #y_train = y_train.astype('float16')
    x_val = images[3100:5000]
    y_val = masks[3100:5000]
#    x_val, y_val = get_imp_nparr(x_val, y_val)
#    x_val = x_val.astype('float16')
#    y_val = y_val.astype('float16')
    model = get_unet0()
    nb_epoch = 200
    history = History()
    callbacks = [
        history,
    ]
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
     
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['mse'])
     
    print(model.summary())
    checkpointer =  keras.callbacks.ModelCheckpoint(filepath="unet_random_crop_weights.hdf5", verbose=1, save_best_only=True)
    model_hist = model.fit(x=x_train, y=y_train, batch_size=32, epochs=nb_epoch, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpointer, reduce_lr], shuffle=True)
    model.save('unet_random_crop.h5')
    model_loss = model_hist.history['loss']
    model_acc = model_hist.history['mean_squared_error']
    val_loss = model_hist.history['val_loss']
    val_acc = model_hist.history['val_mean_squared_error']
    epochs = range(1, nb_epoch+1)
    f_loss = 'curve_loss_deep_unet_random_crop.png'
    f_acc = 'curve_acc_deep_unet_random_crop.png'
    if (os.path.isfile(f_loss) == False):
        plt.plot(epochs, model_loss, 'b+', label='Loss')
        plt.plot(epochs, val_loss, 'r+', label='Val_Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Training/Val Loss')
        plt.legend()
        plt.show()
#        plt.savefig(f_loss)
    if(os.path.isfile(f_acc) == False):
        plt.plot(epochs, model_acc, 'bo', label='Acc')
        plt.plot(epochs, val_acc, 'ro', label='Val_Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Training/Val Accuracy')
        plt.legend()
#        plt.savefig(f_acc)
