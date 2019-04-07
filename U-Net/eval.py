from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import os
import sys
import copy
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from UNet import *
from createDataset import MyDataset
from utils import *
from torchvision.utils import make_grid, save_image
import datetime
from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

OUTPUT_DIR = './results/'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


########### Arguments ###########
device = torch.device("cuda")
print(device)
max_epoch = 1
labels = 11
dataroot = '/home/apoorv/Documents/BTP/data/subslice_f'
batch = 1
save_file = '7_1000_model.pth'
img_size = (128,128)

########### Model ###########
model = UNet(labels).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(save_file))
print(model)

########### Transforms ###########
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation = 2),
        transforms.ToTensor(),
])

########### Dataloader ###########
seg_path = 'segment/'
img_path = 'grey/'

colpath = os.path.join(dataroot, img_path)
segpath = os.path.join(dataroot, seg_path)

colimg = os.listdir(colpath)
segimg = os.listdir(segpath)
colimg.sort()
segimg.sort()
		
X_train, X_test, Y_train, Y_test = train_test_split(colimg, segimg, random_state=42)

train_dataset = MyDataset(X_train, Y_train, dataroot, in_transforms = input_transforms, size = img_size,
	phase = 'train')
test_dataset = MyDataset(X_test, Y_test, dataroot, in_transforms = input_transforms, size = img_size,
	phase = 'train')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=True)

########### Begin Testing ###########
epoch = 0
while(epoch < max_epoch):
	for j in ['train']:
		model.eval()
		dataloader = test_dataloader
		for i,data in enumerate(dataloader):
			img = data[0].to(device).float()
			seg = data[1].to(device)
			truth = seg.cpu().numpy()
			truth = truth[0,:,:]

			output = model(img)
			
			test = torch.max(output.data, 1)[1]
			test = test.cpu().numpy()
			test = test[0,:,:]
			plt.imsave(OUTPUT_DIR+'{}_{}.png'.format(i, 'output'), test)
			plt.imsave(OUTPUT_DIR+'{}_{}.png'.format(i, 'truth'), truth)
	epoch+=1




