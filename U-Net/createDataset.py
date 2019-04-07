import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import scipy.misc as smi
import sys


class MyDataset(data.Dataset):
	def __init__(self, X, Y, root, in_transforms, size, phase, num_classes = 10):
		seg_path = '/masks/'
		img_path = '/imagery/'
		
		self.labels = num_classes
		self.size = size
		self.in_transforms = in_transforms
		
		# img_path = img_path+phase
		# seg_path = seg_path+phase
		
		self.colpath = root + img_path
		self.segpath = root + seg_path
		# self.colpath = os.path.join(root, img_path)
		# self.segpath = os.path.join(root, seg_path)
		# print(self.colpath)
		# print(self.segpath)
		# self.colimg = os.listdir(self.colpath)
		# self.segimg = os.listdir(self.segpath)
		self.root = root
		self.X = X
		self.Y = Y
	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		# print(os.path.join(self.colpath, self.X[index]))
		img = Image.open(os.path.join(self.colpath, self.X[index]))
		img = self.in_transforms(img)

		seg = smi.imread(os.path.join(self.segpath, self.Y[index]))
		seg = smi.imresize(seg, self.size)
		seg = torch.from_numpy(seg)
		seg = seg.long()
		seg[seg>0] = 1
		# print(seg.max(), seg.min(), img.max(), img.min())
		return img, seg
