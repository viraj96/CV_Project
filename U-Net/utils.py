import numpy as np
import torch
import torch.nn as nn
from skimage import img_as_float
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve


def metrics(y_true, predicted, LABELS):
	precision, recall, fscore, support = score(y_true, predicted, labels = LABELS)
	print(len(precision), len(recall), len(fscore), len(support))
	return precision, recall, fscore, support

def checkAccuracy(pred, truth, batch_size):
	pred = pred.cpu().numpy()
	# print(np.unique(pred))
	truth = truth.cpu().numpy()
	acc = np.count_nonzero(pred==truth) / (128*128*batch_size)
	return acc

def checkiou(pred, truth, batch_size):
	intersection = pred & truth
	union = pred | truth
	iou = torch.mean((torch.sum(intersection).float()/torch.sum(union).float()).float()) 
	return iou

def transform_batch(transform, batch_images):
	t = []
	for img in batch_images:
		print(img.shape, img.type)

		img = torch.from_numpy(img_as_float(img.numpy()))
		print(img.max(), img.min())
		img_t = transform(img.repeat(1,3,1,1).permute(1,2,0).numpy())
		print(img_t.shape, img_t.max(), img_t.min())
		exit()
		t.append(img_t)
		# torch.cat((t,img_t), 0)

	# t = np.asarray(t) 
	return torch.stack(t)

def initialize_weights(*models):
	for model in models:
		for module in model.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()
