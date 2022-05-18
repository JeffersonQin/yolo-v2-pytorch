import cv2
import torch
import torchvision
from PIL import Image
from typing import Optional
from matplotlib import pyplot as plt
import numpy
from . import globalvar as G


def draw_precision_recall(pr_data: list, class_idx: Optional[int]=None):
	"""Draw Precision-Recall Curve

	Args:
		pr_data (list): Precision Recall Curve Data
		class_idx (Optional[int]): Class index, used to render title
	"""
	p = [data['precision'] for data in pr_data]
	r = [data['recall'] for data in pr_data]

	plt.plot(r, p, 'o-', color='r')
	plt.xlabel("Recall")
	plt.ylabel("Precision")

	if class_idx is not None:
		plt.title(G.get('categories')[class_idx])

	plt.show()


def draw_detection_result(img: numpy.ndarray, bbox: torch.Tensor, thres: float=0.1) -> numpy.ndarray:
	"""
	Tool function to draw detection result on image
	:param img (numpy.ndarray): numpy image to be rendered
	:param bbox (torch.Tensor): 
	:param thres (float): threshold to filter out low confidence boxes
	:return (numpy.ndarray): image with detection result
	"""
	for i in range(bbox.shape[0]):
		x1, y1, x2, y2, iou = bbox[i][0:5]
		score, cat = bbox[i][5:(5 + G.get('num_classes'))].max(dim=0)
		if iou * score < thres: continue
		img = draw_box(img, float(x1), float(y1), float(x2), float(y2), float(score * iou), G.get('categories')[cat], G.get('colors')[cat])
	return img


def draw_box(img: numpy.ndarray, x1: float, y1: float, x2: float, y2: float, score: float, category: str, color: list):
	"""
	Tool function to draw confidence box on image
	:param x1, y1, x2, y2 (float): coordinates
	:category (str): category
	:color (list): color (255, 255, 255)
	"""
	x1 = x1 * img.shape[1]
	x2 = x2 * img.shape[1]
	y1 = y1 * img.shape[0]
	y2 = y2 * img.shape[0]

	text = category + " " + str(score)
	cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

	text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
	p1 = (int(x1), int(y1) - text_size[1])

	cv2.rectangle(img,
		(p1[0] - 2//2, p1[1] - 2 - baseline),
		(p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
	cv2.putText(img, text,
		(p1[0], p1[1] + baseline),
		cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
	
	return img


def tensor_to_PIL(img):
	"""Convert a tensor into a PIL image"""
	to_pil = torchvision.transforms.ToPILImage()
	return to_pil(img.cpu()).convert('RGB')


def tensor_to_cv2(img):
	return PIL_to_cv2(tensor_to_PIL(img))


def PIL_to_cv2(img):
	"""
	Tool function to convert PIL image to cv2 image
	:param img: PIL image
	:return: cv2 image
	"""
	img = numpy.array(img)
	img = img[:, :, ::-1].copy()
	return img


def cv2_to_PIL(img):
	"""
	Tool function to convert cv2 image to PIL image
	:param img: cv2 image
	:return: PIL image
	"""
	img = img[:, :, ::-1].copy()
	img = Image.fromarray(img)
	return img
