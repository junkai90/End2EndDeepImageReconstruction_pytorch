'''
End-to-end deep image reconstruction (Pytorch): test script
Author: Jun Kai Ho
Date: 2021-7-21
'''

import os
import time
import argparse
from glob import glob

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import pandas as pd
from PIL import Image

import model

import bdpy
import PIL
import scipy.io as sio

from utils import image_deprocess, normalise_img, clip_extreme_value


# Parameters
gen_file = 'snapshots/net_g_10000.pth'
train_file = './data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5'
test_file = './data/fmri/sub-01_perceptionNaturalImageTest_original_VC.h5'
save_dir = 'recon_results/'
mapper_file = 'stimulus_ImageNetTest.tsv'
img_file_suffix = '.JPEG'
ROI_selector_str = 'ROI_VC = 1'
scale = 4.9

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device('cuda')


mapper_df = pd.read_csv(mapper_file, sep='\t', header=None)
mapper_df = mapper_df[[1,0]]
mapper = dict(mapper_df.values)

# response
train_bdata = bdpy.BData(train_file)
train_fmri_data = train_bdata.select(ROI_selector_str)
fmri_data_mean = train_fmri_data.mean(axis=0)
fmri_data_std = train_fmri_data.std(axis=0, ddof=1)


bdata = bdpy.BData(test_file)
test_fmri_data = bdata.select(ROI_selector_str)
test_fmri_labels = bdata.select('image_index').flatten()

# Average test fMRI data across the trials corresponding to the same stimulus image
test_fmri_data_avg = []
img_file_name_list = []
unique_labels = np.unique(test_fmri_labels)
for label in unique_labels:
    sample_index = test_fmri_labels==label
    test_fmri_data_sample = test_fmri_data[sample_index,:]
    test_fmri_data_avg.append(np.mean(test_fmri_data_sample, axis=0))
    
    img_file_name = mapper[label]  # Convet fMRI labels from float ('stimulus_id') to file name labes (str)
    img_file_name_list.append(img_file_name)
    
test_fmri_data_avg = np.vstack(test_fmri_data_avg)
num_of_sample = test_fmri_data_avg.shape[0]

test_fmri_data_avg = (test_fmri_data_avg - fmri_data_mean) / fmri_data_std
test_fmri_data_avg = test_fmri_data_avg * scale


net = model.Generator(input_size=test_fmri_data_avg.shape[1])
net.load_state_dict(torch.load(gen_file))
net.to(device)
net.eval()

# main
for img_index in range(num_of_sample):
    #
    print('img_index='+str(img_index))
    
    # input data
    input_data = torch.tensor(test_fmri_data_avg[img_index,:].astype(np.float32)).to(device)
    input_data = input_data.unsqueeze(0)
    
    # gen img
    gen_img = net(input_data)

    #gen_img = tfs.functional.center_crop(gen_img, (227,227))
    gen_img = gen_img.detach().cpu().numpy()
    
    gen_img = image_deprocess(gen_img)
    
    # save
    save_name = img_file_name_list[img_index] + '.mat'
    sio.savemat(os.path.join(save_dir,save_name),{'gen_img':gen_img})
    
    # To better display the image, clip pixels with extreme values (0.02% of
    # pixels with extreme low values and 0.02% of the pixels with extreme high
    # values). And then normalise the image by mapping the pixel value to be
    # within [0,255]
    save_name = img_file_name_list[img_index] + '.tif'
    PIL.Image.fromarray(np.transpose(normalise_img(clip_extreme_value(gen_img, pct=4)),(1,2,0))).save(os.path.join(save_dir,save_name))
    

##
print('done!')
