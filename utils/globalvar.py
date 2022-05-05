import json
import random
import torch


def init(S=13, B=5):
	"""Init the global variables"""
	global global_dict
	global_dict = {}
	# init values
	global_dict['categories'] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
	global_dict['num_classes'] = len(global_dict['categories'])
	global_dict['colors'] = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(global_dict['num_classes'])]
	global_dict['S'] = S
	global_dict['B'] = B
	# init anchors
	with open(f'anchors/anchors-{B}.json', 'r', encoding='utf-8') as f:
		anchors = json.load(f)
	global_dict['anchors'] = torch.zeros(len(anchors), 2)
	for i in range(len(anchors)):
		global_dict['anchors'][i][0] = anchors[i]['width']
		global_dict['anchors'][i][1] = anchors[i]['height']


def set(key, val):
	"""Set value

	Args:
		key (Any): key
		val (Any): value
	"""
	global_dict[key] = val


def get(key):
	"""Get value

	Args:
		key (Any): key

	Returns:
		Any: value
	"""
	return global_dict[key]
