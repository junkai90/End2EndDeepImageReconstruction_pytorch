'''
End-to-end deep image reconstruction (Pytorch): training script
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
from PIL import Image

import model

import bdpy

from utils import CustomDataset
import torchvision.models as models

# params
parser = argparse.ArgumentParser()
parser.add_argument('--target_layer', type=str, default='12')
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--lambda_feat', default=100, type=float)
parser.add_argument('--lambda_adv', default=100, type=float)
parser.add_argument('--lambda_img', default=1000, type=float)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lr_decay', default=0.5, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--gan_type', default='gan', type=str)
parser.add_argument('--optim_type', default='Adam', type=str)
parser.add_argument('--save_every', default=500, type=int)
params = parser.parse_args()

img_dir = './data/images/train'
resp_dir = './data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5'
save_dir = 'snapshots'
mapper_file = 'stimulus_ImageNetTraining.tsv'
img_file_suffix = '.JPEG'
ROI_selector_str = 'ROI_VC = 1'
device_name = 'cuda:0'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device(device_name)

# response
bdata = bdpy.BData(resp_dir)
fmri_data = bdata.select(ROI_selector_str)
fmri_mean = fmri_data.mean(axis=0)
fmri_norm = fmri_data.std(axis=0, ddof=1)
fmri_data = (fmri_data - fmri_mean) / fmri_norm
fmri_label = bdata.select('image_index').flatten()

# mapper 
mapper_df = pd.read_csv(mapper_file, sep='\t', header=None)
mapper_df = mapper_df[[1,0]]
mapper = dict(mapper_df.values)

# image
transform = transforms.Compose([
            transforms.Resize((248,248)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

img_filenames = [mapper[label] + img_file_suffix for label in fmri_label]
print('Number of images: {}'.format(len(img_filenames)))

img_data = []
for img_ in img_filenames:
    img = Image.open(os.path.join(img_dir, img_))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = transform(img)
    img_data.append(img)
img_data = torch.stack(img_data)

# data
train_dataset = CustomDataset(fmri_data, img_data, transform=transforms.RandomCrop((227,227)))
dataloader = DataLoader(train_dataset, batch_size=params.batch, shuffle=True)


# model
net_g = model.Generator(input_size=fmri_data.shape[1])
net_g.to(device)
net_d = model.Discriminator()
net_d.to(device)
encoder = models.alexnet(pretrained=True)
encoder.to(device)
net_c = model.Comparator(encoder, params.target_layer)
net_c.to(device)
for param in encoder.parameters():
    param.requires_grad = False
for param in net_c.parameters():
    param.requires_grad = False

# optimizer
if params.optim_type == 'Adam':
    optim_g = optim.Adam(net_g.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=0.0004)
    optim_d = optim.Adam(net_d.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=0.0004)
elif params.optim_type == 'RMSprop':
    optim_g = optim.RMSprop(net_g.parameters(), lr=params.lr, alpha=params.beta1)
    optim_d = optim.RMSprop(net_d.parameters(), lr=params.lr, alpha=params.beta1)

loss_func_img = nn.MSELoss()
loss_func_img.to(device)
loss_func_feat = nn.MSELoss()
loss_func_feat.to(device)

lambda_adv = params.lambda_adv
lambda_feat = params.lambda_feat
lambda_img = params.lambda_img


train_discr = True
train_gen = True
for epoch in range(params.epochs+1,1):
    start = time.time()

    net_c.eval()

    std = (1 / 256.) * (1 - epoch / 256.)
    loss_avg = [0, 0, 0, 0, 0]



    for batch_idx, (response, real_image) in enumerate(dataloader):
        response = response.type('torch.FloatTensor').to(device)
        real_image = real_image.type('torch.FloatTensor').to(device)

        # update Generator
        optim_g.zero_grad()
        if train_gen:
            net_g.train()
        net_d.eval()

        
        fake_image = net_g(response)
        fake_logit = net_d(fake_image, std)
        fake_feature = net_c(fake_image)
        real_feature = net_c(real_image)

        loss_feat = loss_func_feat(real_feature, fake_feature)
        loss_img = loss_func_img(real_image, fake_image)

        if params.gan_type == 'lsgan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]

            loss_func_adv = nn.MSELoss()
            loss_func_adv.to(device)
            loss_adv = loss_func_adv(fake_logit, ones)
        elif params.gan_type == 'gan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]

            loss_func_adv = nn.BCEWithLogitsLoss()
            loss_func_adv.to(device)
            loss_adv = loss_func_adv(fake_logit, ones)

        loss_g = lambda_feat * loss_feat + lambda_adv * loss_adv + lambda_img * loss_img

        if train_gen:
            loss_g.backward()
            optim_g.step()

        # update Discriminator
        optim_d.zero_grad()
        net_g.eval()
        if train_discr:
            net_d.train()

        fake_logit = net_d(fake_image.detach(), std)
        real_logit = net_d(real_image, std)

        if params.gan_type == 'lsgan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False, device=device)[np.ones(real_image.shape[0])]
            zeros = torch.eye(2, dtype=torch.float32, requires_grad=False, device=device)[np.zeros(real_image.shape[0])]

            loss_func_adv = nn.MSELoss()
            loss_func_adv.to(device)
            loss_d_real = loss_func_adv(real_logit, ones)*lambda_adv
            loss_d_fake = loss_func_adv(fake_logit, zeros)*lambda_adv
            loss_d = loss_d_real + loss_d_fake
        elif params.gan_type == 'gan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False, device=device)[np.ones(real_image.shape[0])]
            zeros = torch.eye(2, dtype=torch.float32, requires_grad=False, device=device)[np.zeros(real_image.shape[0])]

            loss_func_adv = nn.BCEWithLogitsLoss()
            loss_func_adv.to(device)
            loss_d_real = loss_func_adv(real_logit, ones)*lambda_adv
            loss_d_fake = loss_func_adv(fake_logit, zeros)*lambda_adv
            loss_d = loss_d_real + loss_d_fake

        if train_discr:
            loss_d.backward()
            optim_d.step()

        loss_list = [loss_img*lambda_img, loss_feat*lambda_feat, loss_adv*lambda_adv, loss_d_real, loss_d_fake]
        for i, _loss_ in enumerate(loss_list):
            loss_avg[i] += _loss_.item()

    for i in range(len(loss_avg)):
        loss_avg[i] /= (batch_idx + 1)


    #################
    print('Epoch {}, Run time {:.2f}'.format(epoch, time.time()-start))
    print("recon loss: %f" % (loss_avg[0]))
    print("feat loss: %f" % (loss_avg[1]))
    print("discr real loss: %f" % (loss_avg[3]))
    print("discr fake loss: %f" % (loss_avg[4]))
    print("discr fake loss for generator: %f" % (loss_avg[2]))

    if epoch == 3000:
        params.lr *= params.lr_decay
    elif epoch == 5000:
        params.lr *= params.lr_decay
    elif epoch == 7000:
        params.lr *= params.lr_decay
    elif epoch == 9000:
        params.lr *= params.lr_decay

    #switch optimizing discriminator and generator, so that neither of them overfits too much
    discr_loss_ratio = (loss_avg[3]+loss_avg[4])/ loss_avg[2]
    if discr_loss_ratio < 1e-1 and train_discr:    
        train_discr = False
        train_gen = True
        print("<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (loss_avg[3], loss_avg[4], loss_avg[2], train_discr, train_gen))
    if discr_loss_ratio > 5e-1 and not train_discr:    
        train_discr = True
        train_gen = True
        print(" <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (loss_avg[3], loss_avg[4], loss_avg[2], train_discr, train_gen))
    if discr_loss_ratio > 1e1 and train_gen:
        train_gen = False
        train_discr = True
        print("<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>" % (loss_avg[3], loss_avg[4], loss_avg[2], train_discr, train_gen))
  
    if epoch % params.save_every == 0:
        print('Save checkpoint: {}'.format(epoch))
        torch.save(net_g.state_dict(), save_dir + 'net_g_{}.pth'.format(epoch))
