import random
import math
import warnings
import torch
import torchvision
from torch.utils import data
from . import globalvar as G


__all__ = ['VOCDataset', 'load_data_voc']


class VOCDataset(data.Dataset):
	"""VOC Dataset"""
	
	def __init__(self, dataset: data.Dataset, train=True):
		"""VOCDataset Initialization

		Args:
			dataset (data.Dataset): dataset
			train (bool, optional): whether is used for training. Defaults to True.
		"""
		self.dataset = dataset
		self.train = train


	def __len__(self):
		"""Get length of dataset

		Returns:
			int: length of dataset
		"""
		return len(self.dataset)


	def __getitem__(self, idx):
		"""Get item from dataset

		Args:
			idx (int): index of dataset
		
		Returns:
			tuple: (image, target)
		"""
		img, target = self.dataset[idx]

		if not isinstance(target['annotation']['object'], list):
			target['annotation']['object'] = [target['annotation']['object']]
		count = len(target['annotation']['object'])
		height, width = int(target['annotation']['size']['height']), int(target['annotation']['size']['width'])

		S = G.get('S')
		B = G.get('B')
		num_classes = G.get('num_classes')

		# Image Augmentation
		if self.train:
			# randomly scaling and translation up to 20%
			if random.random() < 0.5:
				# use random value to decide scaling factor on x and y axis
				random_height = random.random() * 0.2
				random_width = random.random() * 0.2
				# use random value again to decide scaling factor for 4 borders
				random_top = random.random() * random_height
				random_left = random.random() * random_width
				# calculate new width and height and position
				top = random_top * height
				left = random_left * width
				height = height - random_height * height
				width = width - random_width * width
				# crop image
				img = torchvision.transforms.functional.crop(img, int(top), int(left), int(height), int(width))
			
				# update target
				for i in range(count):
					obj = target['annotation']['object'][i]
					obj['bndbox']['xmin'] = max(0, float(obj['bndbox']['xmin']) - left)
					obj['bndbox']['ymin'] = max(0, float(obj['bndbox']['ymin']) - top)
					obj['bndbox']['xmax'] = min(width, float(obj['bndbox']['xmax']) - left)
					obj['bndbox']['ymax'] = min(height, float(obj['bndbox']['ymax']) - top)
			
			# adjust saturation randomly up to 150%
			if random.random() < 0.5:
				random_saturation = random.random() + 0.5
				img = torchvision.transforms.functional.adjust_saturation(img, random_saturation)

		# resize
		img = torchvision.transforms.functional.resize(img, (S * 32, S * 32))

		# update labels from absolute to relative
		height, width = float(height), float(width)
		for i in range(count):
			obj = target['annotation']['object'][i]
			obj['bndbox']['xmin'] = float(obj['bndbox']['xmin']) / width
			obj['bndbox']['ymin'] = float(obj['bndbox']['ymin']) / height
			obj['bndbox']['xmax'] = float(obj['bndbox']['xmax']) / width
			obj['bndbox']['ymax'] = float(obj['bndbox']['ymax']) / height

		# Label Encoding
		# [{'name': '', 'xmin': '', 'ymin': '', 'xmax': '', 'ymax': '', }, {}, {}, ...]
		# ==>
		# [x, y  (relative to cell), width, height, 1 if exist (confidence * IoU), one-hot encoding of 20 categories
		#  ... (repeat for each object)]
		label = torch.zeros((S, S, B * (5 + num_classes)))
		obj_cnt = torch.zeros((S, S))
		for i in range(count):
			obj = target['annotation']['object'][i]
			xmin = obj['bndbox']['xmin']
			ymin = obj['bndbox']['ymin']
			xmax = obj['bndbox']['xmax']
			ymax = obj['bndbox']['ymax']
			name = obj['name']
			difficult = (obj['difficult'] == '1')
			if difficult:
				iou = 1.0000001
			else:
				iou = 1.0

			if xmin == xmax or ymin == ymax:
				continue
			if xmin >= 1 or ymin >= 1 or xmax <= 0 or ymax <= 0:
				continue

			x = (xmin + xmax) / 2
			y = (ymin + ymax) / 2

			w = xmax - xmin
			h = ymax - ymin

			xidx = math.floor(x * S)
			yidx = math.floor(y * S)

			if obj_cnt[yidx][xidx] > 4:
				warnings.warn(f'More than {B} objects in one cell ({S}x{S}): {target["annotation"]["folder"]}/{target["annotation"]["filename"]}', RuntimeWarning, stacklevel=2)
				continue

			label[yidx][xidx][int(0 + (5 + num_classes) * obj_cnt[yidx][xidx])] = x * S - xidx
			label[yidx][xidx][int(1 + (5 + num_classes) * obj_cnt[yidx][xidx])] = y * S - yidx
			label[yidx][xidx][int(2 + (5 + num_classes) * obj_cnt[yidx][xidx])] = w
			label[yidx][xidx][int(3 + (5 + num_classes) * obj_cnt[yidx][xidx])] = h
			label[yidx][xidx][int(4 + (5 + num_classes) * obj_cnt[yidx][xidx])] = iou
			label[yidx][xidx][int(5 + (5 + num_classes) * obj_cnt[yidx][xidx] + G.get('categories').index(name))] = 1

			obj_cnt[yidx][xidx] += 1

		return img, label


def load_data_voc(batch_size: int, num_workers=0, persistent_workers=False, download=False, train_shuffle=True, test_shuffule=False, pin_memory=True, data_augmentation=True) -> list[data.DataLoader]:
	"""Load Pascal VOC dataset, consist of VOC2007trainval+test+VOC2012train, VOC2012val

	Args:
		batch_size (int): batch size
		num_workers (int, optional): number of workers. Defaults to 0.
		persistent_workers (bool, optional): persistent_workers. Defaults to False.
		download (bool, optional): whether to download. Defaults to False.
		train_shuffle (bool, optional): whether to shuffle train data. Defaults to True.
		test_shuffule (bool, optional): whether to shuffle test data. Defaults to False.
		pin_memory (bool, optional): whether to pin memory. Defaults to True.

	Returns:
		list[data.DataLoader]: train_iter, test_iter
	"""
	trans = [ torchvision.transforms.ToTensor() ]
	trans = torchvision.transforms.Compose(trans)

	voc2007_trainval = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=download, transform=trans)
	voc2007_test = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='test', download=download, transform=trans)
	voc2012_train = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='train', download=download, transform=trans)
	voc2012_val = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='val', download=download, transform=trans)

	return (
		data.DataLoader(VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), train=data_augmentation), 
			batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory),
		data.DataLoader(VOCDataset(voc2012_val, train=False),
			batch_size=batch_size, shuffle=test_shuffule, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
	)
