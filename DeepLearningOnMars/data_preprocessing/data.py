import numpy as np
from PIL import Image
import os
import cv2
import tifffile as tiff
import pandas as pd
import h5py
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from math import floor
from random import randint
import argparse
 
parser = argparse.ArgumentParser(description='Data Preprocessing & Data Stats')
parser.add_argument('-crop_size', metavar='CS', type=int, default=128, help='An integer for the crop size.')
parser.add_argument('--stats', action='store_true', help='Run stats mode')
args = parser.parse_args()
 
Image.MAX_IMAGE_PIXELS = 10000000000
 
def get_ids():
    l = []
    for folder in os.listdir('IMPACTS'):
        if os.path.isdir('IMPACTS_GT/'+folder):
            for files in os.listdir('IMPACTS/'+folder):
                if (files[-2] == 'i'):
                    l.append((folder, files[0: -8]))
    return l
 
"""
Return the numpy arrays for the image and the mask relative to the data_id in the folder_id
"""
def read_data(folder_id, data_id):
    x = [folder_id, data_id]
    img = Image.open('IMPACTS/{}/{}.geo.tif'.format(folder_id, data_id))
    mask = Image.open('IMPACTS_GT/{}/{}_gt.tif'.format(folder_id, data_id))
    np_img = np.fromstring(img.tobytes(), dtype=np.uint8)
    np_mask = np.fromstring(mask.tobytes(), dtype=np.uint8)
    return np_img.reshape((img.size[1], img.size[0])), np_mask.reshape((mask.size[1], mask.size[0]))
 
def crop_image(folder_id, data_id, crop_size=1024):
    image, mask = read_data(folder_id, data_id)
    x, y = image.shape
    n = int(floor(x/crop_size))
    m = int(floor(y/crop_size))
    images = np.array([])
    masks = np.array([])
    for i in tqdm(range(n)):
        for j in range(m):
            im = image[i:(i+crop_size), j:(j+crop_size)]
            msk = mask[i:(i+crop_size), j:(j+crop_size)]
            images = np.append(images, im)
            masks = np.append(masks, msk)
    if K.image_data_format() == 'channels_first':
        return images.reshape((1, n*m, crop_size, crop_size)), masks.reshape((1, n*m, crop_size, crop_size))
    else:
        return images.reshape((n*m, crop_size, crop_size, 1)), masks.reshape((n*m, crop_size, crop_size, 1))
 
def crop_impact_area(crop_size):
    ids = get_ids()
    images = np.array([])
    masks = np.array([])
    print('croping impact areas of', crop_size, 'x', crop_size)
    nb_imp = 0
    for i in tqdm(ids):
        img, msk = read_data(i[0], i[1])
        x_mask, y_mask = np.nonzero(msk)
        if (x_mask.size > 0):
            x_imp = x_mask[0]
            y_imp = y_mask[0]
            if ((x_imp + crop_size < img.shape[0])
                & (x_imp - crop_size > 0)
                & (y_imp - crop_size > 0)
                & (y_imp + crop_size < img.shape[1])):
                rx_left = randint(0, crop_size)
                rx_right = crop_size - rx_left
                ry_up = randint(0, crop_size)
                image = img[int(x_imp - crop_size/2) : int(x_imp + (crop_size/2)), int(y_imp - (crop_size)/2): int(y_imp + (crop_size/2))]
                mask =  msk[int(x_imp - (crop_size/2)) : int(x_imp + (crop_size/2)), int(y_imp - (crop_size/2)): int(y_imp + (crop_size/2))]
                images = np.append(images, image)
                masks = np.append(masks, mask)
                nb_imp += 1
    if K.image_data_format() == 'channels_first':
        return images.reshape((1, nb_imp, crop_size, crop_size)), masks.reshape((1, nb_imp, crop_size, crop_size))
    else:
        return images.reshape((nb_imp, crop_size, crop_size, 1)), masks.reshape((nb_imp, crop_size, crop_size, 1))
 
"""
Crop an image randomly around an impact such that the impact can be found anywhere in the image.
Need to do so as it prevents the network from learning that impact are always in the same place within an image.
"""
def random_crop_impact_area(crop_size):
    ids = get_ids()
    images = np.array([])
    masks = np.array([])
    nb_imp = 0
    for i in tqdm(ids):
        img, msk = read_data(i[0], i[1])
        x_mask, y_mask = np.nonzero(msk)
        if x_mask.size > 0:
            r_ind = randint(0, x_mask.size -1)
            x_imp = x_mask[r_ind]
            y_imp = y_mask[r_ind]
            if ((x_imp + crop_size < img.shape[0])
                & (x_imp - crop_size > 0)
                & (y_imp - crop_size > 0)
                & (y_imp + crop_size < img.shape[1])):
                for i in range(10):
                    rx_left = randint(0, crop_size)
                    rx_right = crop_size - rx_left
                    ry_up = randint(0, crop_size)
                    ry_down = crop_size - ry_up
                    image = img[int(x_imp - rx_left) : int(x_imp + rx_right), int(y_imp - ry_down): int(y_imp + ry_up)]
                    mask =  msk[int(x_imp - rx_left) : int(x_imp + rx_right), int(y_imp - ry_down): int(y_imp + ry_up)]
                    images = np.append(images, image)
                    masks = np.append(masks, mask)
                    nb_imp += 1
    if K.image_data_format() == 'channels_first':
        return images.reshape((1, nb_imp, crop_size, crop_size)), masks.reshape((1, nb_imp, crop_size, crop_size))
    else:
        return images.reshape((nb_imp, crop_size, crop_size, 1)), masks.reshape((nb_imp, crop_size, crop_size, 1))
 
"""
Return numpy arrays of cropped images and associated masks with no impact from an image and the associated mask.
Network need also to be shown image with no impact since an impact is quite rare.
"""
def get_zeros_matrix(image, mask, crop_size):
    assert(crop_size < mask.shape[0] and crop_size < mask.shape[1])
    m = int(mask.shape[0]/crop_size) - 1
    n = int(mask.shape[1]/crop_size) - 1
    images = np.array([])
    masks = np.array([])
    nb_data = 0
    cpt = 0
    for i in range(m):
        for j in range(n):
            if i + j <= 20:
                msk = mask[int(i*crop_size): int((i+1)*crop_size), int(j*crop_size):  int((j+1)*crop_size)]
                img = image[int(i*crop_size): int((i+1)*crop_size), int(j*crop_size):  int((j+1)*crop_size)]
                x_msk, y_msk = np.nonzero(msk)
                if (x_msk.size == 0):
                    images = np.append(images, img)
                    masks = np.append(masks, msk)
                    nb_data += 1
    if K.image_data_format() == 'channels_first':
        return images.reshape((1, nb_data, crop_size, crop_size)), masks.reshape((1, nb_data, crop_size, crop_size))
    else:
        try:
            return images.reshape((nb_data, crop_size, crop_size, 1)), masks.reshape((nb_data, crop_size, crop_size, 1))
        except:
            return 'Can not reshape the images'
 
"""
Use the previous function to do so on the whole data.
Return images and masks with no impact in numpy arrays
"""
def get_empty_dataset(crop_size):
    ids = get_ids()
    images = np.array([])
    masks = np.array([])
    nb_data = 0
    for i in tqdm(ids):
        image, mask = read_data(i[0], i[1])
        img, msk = get_zeros_matrix(image, mask, crop_size)
        if img.shape[0] > 0:
            images = np.append(images, img)
            masks = np.append(masks, msk)
            nb_data += img.shape[0]
    if K.image_data_format() == 'channels_first':
        return images.reshape((1, nb_data, crop_size, crop_size)), masks.reshape((1, nb_data, crop_size, crop_size))
    else:
        return images.reshape((nb_data, crop_size, crop_size, 1)), masks.reshape((nb_data, crop_size, crop_size, 1))
 
"""
Final function to create random cropped dataset.
Half images and associated masks with impacts and half with no impact
"""
def create_random_crop_dataset(crop_size):
    empty_img, empty_msk = get_empty_dataset(crop_size) #, fp_images, fp_masks = get_empty_dataset(crop_size)
    impacts_img, impacts_msk = random_crop_impact_area(crop_size)
    assert(impacts_img.shape == impacts_msk.shape)
    assert(empty_img.shape  == empty_msk.shape)
    m = min(impacts_img.shape[0], empty_img.shape[0])
    images = np.array([])
    masks = np.array([])
    for i in tqdm(range(m)):
        images = np.append(images, impacts_img[i])
        images = np.append(images, empty_img[i])
        masks = np.append(masks, impacts_msk[i])
        masks = np.append(masks, empty_msk[i])
    return images.reshape((2*m, crop_size, crop_size, 1)), masks.reshape((2*m, crop_size, crop_size, 1))#, fp_images, fp_masks
 
"""
Computes the proportion of all the impacts in the image
"""
def impact_proportion(folder_id, data_id):
    np_mask = read_data(folder_id, data_id)[1]
    x_mask, y_mask = np_mask.shape
    x_imp, y_imp = np.nonzero(np_mask)
    return 100*len(x_imp)/(x_mask * y_mask) #get the result in %
 
"""
Computes the average proportion of impacts in the images 
"""
def stats():
    ids = get_ids()
    cpt = 0
    i = 0
    for data_id in tqdm(ids):
        cpt += 100 * impact_proportion(data_id[0], data_id[1])
        i += 1
        if (i == 100):
            return cpt/i
 
"""
Create a csv file to show the user the heterogeneity of image's dimensions
"""
def tif_to_csv():
    Files = []
    Width = []
    Height = []
    for folder in tqdm(os.listdir('IMPACTS')):
        if (os.path.isdir('IMPACTS_GT/'+folder)):
            for f in os.listdir('IMPACTS/'+folder):
                if (f[-2] == 'i'):
                    image_id = f[0: -8]
                    image = tiff.imread('IMPACTS/'+folder+'/'+f)
                    height, width = image.shape
                    Files += [image_id]
                    Width += [width]
                    Height += [height]
    df = pd.DataFrame({'file_name':Files, 'width':Width, 'height':Height})
    df.to_csv('../data/data.csv', index=False)
 
if __name__ == "__main__":
    if args.stats:
        stats()
        if (os.path.isfile('data/data.csv') == False):
            tif_to_csv()
    cs = args.crop_size
    images, masks = create_random_crop_dataset(cs) 
    np.save('random_cropped_images_{}'.format(cs), images)
    np.save('random_cropped_masks_{}'.format(cs), masks)
