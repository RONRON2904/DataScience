from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.append('./data_preprocessing')
from data import get_empty_nparr
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def predict(img_input, model, crop_size):
    if (os.path.isfile(model) == False):
        exit(0)
        print('No model found')
    img_input = img_input.reshape(img_input.shape[0], crop_size, crop_size, 1)
    m = load_model(model)
    return m.predict(img_input)

def postprocessing(predicted_mask, threshold):
    k, n, m, s = predicted_mask.shape
    for t in range(k):
        for i in range(n):
            for j in range(m):
                if (predicted_mask[t, i, j, 0] <= threshold):
                    predicted_mask[t, i, j, 0] = 0
                else:
                    predicted_mask[t, i, j, 0] = 1
    return predicted_mask

def diff(predicted_mask, true_mask):
    d = true_mask - predicted_mask
    return np.nonzero(d)[0].size/(true_mask.shape[0]*true_mask.shape[1])
            
def plot_impact(img, mask, name):
    img = np.dstack([img.astype(np.uint8)]*3)
    heatmap = np.uint8(255*mask)
    heatmap = heatmap[:, :, None]
    heatmap = np.dstack((np.zeros_like(heatmap), heatmap, heatmap))
    superposed_img = np.where(heatmap, 255, img)
    cv2.imwrite(name, superposed_img)

def get_ind(mask, predicted_mask):
    inds = np.array([])
    for i in range(mask.shape[0]):
        x, y = np.nonzero(mask[i, :, :])
        if (x.shape[0] == 0 and mse(mask[i], predicted_mask[i]) > 0.001):
            inds = np.append(inds, i)
    return inds

def get_positive_inds(mask):
    inds = np.array([])
    for i in range(mask.shape[0]):
        x, y = np.nonzero(mask[i, :, :])
        if (x.shape[0] > 10):
            inds = np.append(inds, i)
    return inds
    
"""
In order to make the Unet improves, we need to give it false positives, specially those with high mse
Need to have false_positive_img and false positive_msk files before
"""
def get_false_positive(unet, crop_size):
    x = np.load('false_positive_img.npy')
    y = np.load('false_positive_msk.npy')
    fpi = np.array([])
    fpm= np.array([])
    nb_fp = 0
    y_pred = predict(x, unet)
    for i in tqdm(range(x.shape[0])):
        if mse(y_pred[i, :, :, 0], y[i, :, :, 0]) > 0.05:
            fpi = np.append(fpi, x[i])
            fpm = np.append(fpm, y[i])
            nb_fp += 1
    return fpi.reshape((nb_fp, crop_size, crop_size, 1)), fpm.reshape((nb_fp, crop_size, crop_size, 1))
    
"""
It compputes the average mse made by a network on the whole test data 
"""
def bench(unet, images_npfile, masks_npfile, crop_size):
    x_tst = np.load(images_npfile)
    y_tst = np.load(masks_npfile)
    x_tst = x_tst[5000:6020]       #these number have to be changed depending on the data preprocessing before
    masks = y_tst[5000:6020]          # same here
    masks = masks.reshape(masks.shape[0], crop_size, crop_size)
    predicted_masks = predict(x_tst, unet)
    predicted_masks = predicted_masks.reshape(predicted_masks.shape[0], crop_size, crop_size)
    total_mse = 0
    for i in range(masks.shape[0]):
        total_mse += mse(predicted_masks[i], masks[i])
    return total_mse/masks.shape[0]
            
if __name__ == '__main__':
    x = np.load('data/images.npy') #file needs to be changed
    y = np.load('data/masks.npy')  #same here
    x_test = x[5000:6020]    #numbers need to be changed 
    mask = y5000:6020]       #same here
    crop_size = 128
    print(bench("NETs/unet1.h5"))
    print(bench("NETs/unet2.h5"))
    print(bench("NETs/unet3.h5"))
    print(bench("NETs/unet_random_crop_weights.hdf5"))
    
    predicted_mask1 = predict(x_test, "NETs/unet1.h5")
    predicted_mask1 = predicted_mask1.reshape(predicted_mask1.shape[0], crop_size, crop_size)
    predicted_mask2 = predict(x_test, "NETs/unet2.h5")
    predicted_mask2 = predicted_mask2.reshape(predicted_mask2.shape[0], crop_size, crop_size)
    predicted_mask3 = predict(x_test, "NETs/unet3.h5")
    predicted_mask3 = predicted_mask3.reshape(predicted_mask3.shape[0], crop_size, crop_size)
    predicted_mask4 = predict(x_test, "NETs/unet_random_crop_weights.hdf5")
    predicted_mask4 = predicted_mask4.reshape(predicted_mask4.shape[0], crop_size, crop_size)
    
    x_test = x_test.reshape(x_test.shape[0], crop_size, crop_size)
    mask = mask.reshape(mask.shape[0], crop_size, crop_size)
    plt.figure(figsize=(15, 4))
    n = 5
    ind = get_positive_inds(mask)
    print(ind.shape)
    for i in range(int(n)):
        indice = ind[10*i]
        ax = plt.subplot(6, n, i+1)
        plt.imshow(x_test[int(indice)], cmap ='viridis')
        ax = plt.subplot(6, n, i+1 + n)
        plt.imshow(mask[int(indice)], cmap ='viridis')
        ax = plt.subplot(6, n, i + 1 + (2*n))
        plt.imshow(predicted_mask1[int(indice)], cmap ='viridis')
        ax = plt.subplot(6, n, i + 1 + (3*n))
        plt.imshow(predicted_mask2[int(indice)], cmap ='viridis')
        ax = plt.subplot(6, n, i + 1 + (4*n))
        plt.imshow(predicted_mask3[int(indice)], cmap ='viridis')
        ax = plt.subplot(6, n, i + 1 + (5*n))
        plt.imshow(predicted_mask4[int(indice)], cmap ='viridis')
         d1 = mse(predicted_mask1[int(indice)], mask[int(indice)])
        d2 = mse(predicted_mask2[int(indice)], mask[int(indice)])
        d3 = mse(predicted_mask3[int(indice)], mask[int(indice)])
        d4 = mse(predicted_mask4[int(indice)], mask[int(indice)])
        print(d1, d2, d3, d4)
        plt.show()
