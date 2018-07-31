import torch
import torch.utils.data as data
import numpy as np

def load_data(dtype='float32'):
    images = np.load('../random_cropped_images.npy')
    masks = np.load('../random_cropped_masks.npy')
    x_train = images[:1000]
    y_train = masks[:1000]
    x_train = x_train.astype(dtype)
    y_train = y_train.astype(dtype)
    x_val = images[1000:2000]
    y_val = masks[1000:2000]
    x_val = x_val.astype(dtype)
    y_val = y_val.astype(dtype)
    train = (x_train, y_train)
    val = (x_val, y_val)
    test = (images[2000:3000].astype(dtype), masks[2000:3000].astype(dtype))
    return train, val, test

class MarsDataset(data.Dataset):
    def __init__(self, train=True, val=False, test=False):
        self.train_data, self.val_data, self.test_data = load_data()

        if train:
            self.x = self.train_data[0]
            self.y = self.train_data[1]
        elif val:
            self.x = self.val_data[0]
            self.y = self.val_data[1]
        elif test:
            self.x = self.test_data[0]
            self.y = self.test_data[1]
            
    def __getitem__(self, idx):
        return self.x[idx].reshape(1, 256, 256), self.y[idx].reshape(1, 256, 256)

    def __len__(self):
        return self.x.shape[0]
