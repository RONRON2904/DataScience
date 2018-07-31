import numpy as np
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
keras.__version__
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, concatenate

from keras import backend as K

import h5py

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy

import datetime
import os

import random

from keras.models import model_from_json
import threading
import itertools
import sys
sys.path.append("data_preprocessing/")
from data import get_imp_nparr
from predict import get_ind
import matplotlib.pyplot as plt

class Tiramisu:

    def __init__(self, num_channels, img_rows, img_cols, num_mask_channels):
        self.mini_create(num_channels, img_rows, img_cols, num_mask_channels)

    def dense_block(self, layers, filters, x):
        for i in range(layers):
            x0 = BatchNormalization()(x)
            #x1 = Activation('relu')(x0)
            x2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_uniform', activation='relu')(x0)
        #    x3 = Dropout(0.2)(x2)
            x = concatenate([x, x2], axis=3)
        return x

    def transition_down(self, filters, x):
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = Conv2D(filters, 1, padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        #x = Dropout(0.2)(x)
        return MaxPooling2D(pool_size=2)(x)

    def transition_up(self, filters, x):
        return Conv2DTranspose(filters, 3, padding='same', strides=(2, 2), kernel_initializer='he_uniform')(x)

    def mini_create(self, img_rows, img_cols, num_channels, num_mask_channels):
        if K.image_data_format() == 'channels_first':
            inputs = Input(shape=(num_channels, img_rows, img_cols))
        else:
            inputs = Input(shape=(img_rows, img_cols, num_channels))

        x = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform')(inputs)
        skip1 = self.dense_block(4, 24, x)
        x = self.transition_down(24, skip1)

        x = self.dense_block(5, 34, x)

        x = concatenate([self.transition_up(12, x), skip1], axis=3)
        x = self.dense_block(4, 42, x)

        y = Conv2D(num_mask_channels, 1, activation='softmax', kernel_initializer='he_uniform')(x)

        return Model(inputs=inputs, outputs=y)

    def create(self, img_rows, img_cols, num_channels, num_mask_channels):
        if K.image_data_format() == 'channels_first':
            inputs = Input(shape=(num_channels, img_rows, img_cols))
        else:
            inputs = Input(shape=(img_rows, img_cols, num_channels))

        x = Conv2D(48, 3, padding='same', kernel_initializer='he_uniform')(inputs)
        skip1 = self.dense_block(4, 112, x)
        x = self.transition_down(112, skip1)

        skip2 = self.dense_block(5, 192, x)
        x = self.transition_down(192, skip2)

        skip3 = self.dense_block(7, 304, x)
        x = self.transition_down(304, skip3)

        skip4 = self.dense_block(10, 464, x)
        x = self.transition_down(464, skip4)

        skip5 = self.dense_block(12, 656, x)
        x = self.transition_down(656, skip5)

        x = self.dense_block(15, 896, x)

        x = concatenate([self.transition_up(1088, x), skip5], axis=3)
        x = self.dense_block(12, 1088, x)

        x = concatenate([self.transition_up(816, x), skip4], axis=3)
        x = self.dense_block(10, 816, x)

        x = concatenate([self.transition_up(578, x), skip3], axis=3)
        x = self.dense_block(7, 578, x)

        x = concatenate([self.transition_up(384, x), skip2], axis=3)
        x = self.dense_block(5, 384, x)

        x = concatenate([self.transition_up(256, x), skip1], axis=3)
        x = self.dense_block(4, 256, x)

        y = Conv2D(num_mask_channels, 1, activation='sigmoid', kernel_initializer='he_uniform')(x)

        return Model(inputs=inputs, outputs=y)



if __name__ == '__main__':
    images = np.load('x_train.npy')
    masks = np.load('y_train.npy')
    x_train = images[:10000]
    y_train = masks[:10000]
    x_train, y_train = get_imp_nparr(x_train, y_train)
    x_val = images[10000:13000]
    y_val = masks[10000:13000]

    img_rows = 128
    img_cols = 128
    num_channels = 1
    num_mask_channels = 1
    t = Tiramisu(img_rows, img_cols, num_channels, num_mask_channels)
    model = t.mini_create(img_rows, img_cols, num_channels, num_mask_channels)
    nb_epoch = 20
    history = History()
    callbacks = [
        history,
    ]

    model.compile(loss='mse', optimizer=Nadam(lr=1e-2), metrics=['acc'])

    print(model.summary())
    print(x_train.shape)
    model_hist = model.fit(x=x_train, y=y_train, batch_size=32, epochs=nb_epoch, verbose=1, validation_data=(x_val, y_val))
    model.save('tiramisu.h5')
    model_loss = model_hist.history['loss']
    model_acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    val_loss = model_hist.history['val_loss']
    epochs = range(1, nb_epoch+1)
    f = 'curve_tiramisu1024_05.png'
    if (os.path.isfile(f) == False):
        plt.plot(epochs, model_loss, 'b+', label='Loss')
        plt.plot(epochs, model_acc, 'bo', label='Acc')
        plt.plot(epochs, val_acc, 'ro', label='Val_Acc')
        plt.plot(epochs, val_loss, 'r+', label='Val_Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss/Acc')
        plt.legend()
        plt.savefig('curve_unet1024_05.png')
        plt.show()
