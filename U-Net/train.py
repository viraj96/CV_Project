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

LABELS = ['0', '1']

LOG_DIR = './logs/'

now = str(datetime.datetime.now())
OUTPUTS_DIR = './outputs/'

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)
if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
OUTPUTS_DIR = OUTPUTS_DIR + now + '/'
if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
if not os.path.exists(LOG_DIR+now):
	os.makedirs(LOG_DIR+now)

summary_writer = SummaryWriter(LOG_DIR + now)
########### Arguments ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
max_epoch = 100
labels = 2
dataroot = '../'
batch = 40
save_after = 1
lr = 1.0e-5
save_file = 'weights.pth'
img_size = (128,128)
momentum = 0.95
weight_decay = 1.0e-4
resume = False
log = "error_log.txt"

########### Model ###########
model = UNet(labels).to(device)
model = nn.DataParallel(model)
if(resume):
    model.load_state_dict(torch.load(save_file))
    f = open(log, "a")
else:
    f = open(log, "w")
print(model)

########### Transforms ###########
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation = 2),
        transforms.ToTensor(),
])
to_tensor =  transforms.Compose([transforms.ToTensor()])


########### Dataloader ###########
seg_path = '/masks/'
img_path = '/imagery/'

#colpath = os.path.join(dataroot, img_path)
#segpath = os.path.join(dataroot, seg_path)
colpath = '../imagery/'
segpath = '../masks/'
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

# print(len(train_dataloader))
# print(len(test_dataloader))

# for x , y in train_dataloader:
# 	print(x.shape, y.shape)
# 	exit()
# import ipdb; ipdb.set_trace()
########### Criterion ###########
optimizer = optim.Adam([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': lr, 'weight_decay': weight_decay}
], betas=(momentum, 0.999))

########### Begin Training ###########
epoch = 0
while(epoch < max_epoch):
	start = time.time()
	f = open(log, 'a')
	running_loss = 0
	acc = []
	# for j in ['train', 'val']:
	for j in ['train']:
		dataloader = train_dataloader
		for i,data in enumerate(dataloader):
			model.train()
			img = data[0].to(device).float()
			seg = data[1].to(device).long()
			# seg = seg.unsqueeze(1)
			

			npseg = seg.cpu().numpy()
			un, weigh = np.unique(npseg, return_counts = True)
			print(un)
			max_weight = np.max(weigh)
			# if(len(weigh)==10):
			# 	weigh = np.insert(weigh, 0, 1)
			# 	total = np.sum(weigh)
			# weigh = torch.from_numpy(weigh).float() / float(total)
			# weigh = weigh.reciprocal()
			# weigh = weigh.to(device)

			# weights = np.zeros((labels,1))
			# for k in range(len(un)):
			# 	weights[un[k]] = max_weight /  weigh[k]

			# weights = torch.from_numpy(weights).float()
			# weights = weights.to(device)
			# print(weights.shape)

			# print(img.shape, seg.shape)
			# print(un)

			optimizer.zero_grad()
			output = model(img)
			# print(output.shape)

			test = torch.max(output.data, 1)[1]
			cur_accuracy = checkAccuracy(test, seg, batch)
			iou = checkiou(test, seg, batch)
			acc.append(cur_accuracy)
			# weight=torch.Tensor([1.0, 0.25,     20.0,    15.0,    15.0,     20.0,      20.0,     1.0,    20.0,     2.0,    0.25])	
			criterion = nn.NLLLoss().cuda()
			# print(output.shape)
			# print(seg.shape)
			# exit()
			# print(F.log_softmax(output, dim = 1).shape)
			loss = criterion(F.log_softmax(output, dim = 1), seg)
			loss.backward()

			optimizer.step()
			running_loss +=loss.item()

			y = seg.view(-1)
			val = output.max(1)[1].view(-1)

			# print(y.shape)
			# print(val.shape)

			# sys.exit()

			batch_precision, batch_recall, batch_fscore, batch_support = metrics(y.cpu().numpy(), val.cpu().numpy(), LABELS)

			summary_writer.add_scalars('Train Precision/', {LABELS[k]:batch_precision[k] for k in range(len(LABELS))})
			summary_writer.add_scalars('Train Recall/', {LABELS[k]:batch_recall[k] for k in range(len(LABELS))})
			summary_writer.add_scalars('Train Fscore/', {LABELS[k]:batch_fscore[k] for k in range(len(LABELS))})
			summary_writer.add_scalars('Train Support/', {LABELS[k]:batch_support[k] for k in range(len(LABELS))})
			
			summary_writer.add_scalar("accuracy", cur_accuracy)
			summary_writer.add_scalar("loss", loss.item())
			summary_writer.add_scalar("iou", iou.item())
			# print(iou.shape, iou.item())
			print('loss:{0:.4f}'.format(loss.item()), "Epoch:{0} Batch:{1}/{2} accuracy:{3:.5f} IOU:{4:.5f}".format(epoch, i, len(dataloader), cur_accuracy, iou.item()))
			# print(j,i)
			if i % 200 == 0 :
				# print(img.max(), img.min())
				# exit()
				save_image(img, OUTPUTS_DIR + '{}_{}_scan.png'.format(epoch, i))
				# print(torch.max(output, 1)[1].shape)
				# print(seg.type, seg)
				save_image(torch.max(output, 1)[1].cpu().long().unsqueeze(1), OUTPUTS_DIR + '{}_{}_mat.png'.format(epoch, i))
				out_img = 1.0 - torch.max(output, 1)[1].cpu().float().unsqueeze(1) / 255.0
				out_img = out_img.repeat(1,3,1,1)
				out_img[:, 0, :, :] = out_img[:, 0, :, :]*0.2
				out_img[:, 1, :, :] = out_img[:, 1, :, :]*0.4
				out_img[:, 2, :, :] = out_img[:, 2, :, :]*0.6
				save_image(out_img, OUTPUTS_DIR + '{}_{}_out.png'.format(epoch, i))
				# print(out_img.max(), out_img.min, out_img.shape, out_img.type)
				# out_img = transform_batch(to_tensor, out_img)
				# print(seg.max(), seg.min())
				seg_img = 1.0 - seg.float().cpu().unsqueeze(1) / 255.0
				seg_img = seg_img.repeat(1,3,1,1)
				seg_img[:, 0, :, :] = seg_img[:, 0, :, :]*0.2
				seg_img[:, 1, :, :] = seg_img[:, 1, :, :]*0.4
				seg_img[:, 2, :, :] = seg_img[:, 2, :, :]*0.6

				# print(seg_img.max(), seg_img.min())
				# seg_img = transform_batch(to_tensor, seg_img)
				# print(seg_img)
				# exit()
				save_image(seg_img, OUTPUTS_DIR + '{}_{}_seg.png'.format(epoch, i))
				# import ipdb; ipdb.set_trace()
				
			if i % 1000 == 0 and i!=0:
				model.eval()
				ious = []
				for k, data in enumerate(test_dataloader):
					x = data[0]
					y = data[1]
					# print(x.shape)
					# print(y.shape)
					with torch.no_grad():
						x = x.to(device)
						y = y.to(device)
						out = model(x)
						out_labels = torch.max(out.data, 1)[1]
						iou = checkiou(out_labels, y, batch)
						ious.append(iou)
						summary_writer.add_scalar('batch test iou', iou.item())
						print('Test Batch:{0}/{1} IOU:{2:.4f}'.format(k, len(test_dataloader), iou.item()))
				avg_iou = np.average(ious)
				summary_writer.add_scalar('avg test iou', avg_iou)
				torch.save(model.state_dict(), '{}_{}_model.pth'.format(epoch, i))
				print('Epoch: {0} TEST AVG IOU: {1:.4f}'.format(i, avg_iou))





	if(epoch%save_after == 0):
		torch.save(model.state_dict(), '{}_model.pth'.format(epoch))

	acc = np.asarray(acc)
	acc = np.mean(acc)

	s = 'Time of complete of epoch '+str(epoch+1)+' = '+str(time.time() - start)
	print(s)
	f.write(s+"\n")
	s = 'Loss after this epoch = '+str(running_loss)+' and average accuracy = '+str(acc)
	f.write(s+"\n")
	print(s)
	epoch+=1
	f.close()



