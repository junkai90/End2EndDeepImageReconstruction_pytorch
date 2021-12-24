import numpy as np
import torch
import torch.nn as nn




# functions
def image_deprocess(img, img_mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
                  img_std=np.array([0.229, 0.224, 0.225], dtype=np.float32), norm=255):
    """convert from Pytorch's input image layout"""
    image = img * np.array([[img_std]]).T
    return np.dstack((image + np.reshape(img_mean, (3, 1, 1)))) * norm 

def normalise_img(img):
    '''Normalize the image.
    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int
    '''
    img = img - img.min()
    if img.max()>0:
        img = img * (255.0/img.max())
    img = np.uint8(img)
    return img

def clip_extreme_value(img, pct=1):
    '''clip the pixels with extreme values'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img = np.clip(img, np.percentile(img, pct/2.),
                  np.percentile(img, 100-pct/2.))
    return img


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transforms=None):
        self.transforms = transforms
        self.data = data
        self.num_samples = len(data)
        self.label = label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transforms:
            out_label = self.transforms(out_label)
        return out_data, out_label