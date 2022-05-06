from functools import cmp_to_key
from enum import Enum
import numpy as np
import torch

from yolo.converter import Yolo2BBox


__all__ = ['InterpolationMethod', 'CalculationMetrics', 'ObjectDetectionMetricsCalculator']


class InterpolationMethod(Enum):
	Interpolation_11 = 1
	Interpolation_101 = 2


class CalculationMetrics():
	def __init__(self, IoU: float, confidence: float, mustbe_FP: bool):
		"""Initialization for `CalculationMetrics`

		Args:
			IoU (float): intersection over union with ground truth
			confidence (float): detection confidence
			mustbe_FP (bool): if there is already another detection having higher IoU with the same ground truth, then this detection must be False Positive
		"""
		self.IoU = IoU
		self.confidence = confidence
		self.mustbe_FP = mustbe_FP


def compare_metrics(metrics1: CalculationMetrics, metrics2: CalculationMetrics):
	if metrics1.confidence == metrics2.confidence:
		return metrics2.IoU - metrics1.IoU
	return metrics2.confidence - metrics1.confidence


class ObjectDetectionMetricsCalculator():
	"""data
	[   # classes
		{
			"data": [<CalculationMetrics>],
			"detection": <int>,
			"truth": <int>
		}
	]
	"""

	def __init__(self, num_classes: int, confidence_thres: float):
		"""ObjectDetectionMetricsCalculator Initialization

		Args:
			num_classes (int): number of classes detector can classify
			confidence_thres (float): confidence threshold. if the detection's confidence is smaller than the threshold, it would not be counted as a detection. In other words, it would be neither TP nor FP.
		"""
		# initialize data
		self.data = [{"data": [], "detection": 0, "truth": 0} for _ in range(num_classes)]
		self.confidence_thres = confidence_thres


	def _add_data(self, pred: torch.Tensor, truth: torch.Tensor):
		"""Add single image data

		Args:
			pred (torch.Tensor): detection prediction (S, S, 25B)
			truth (torch.Tensor): ground truth (YOLO v2 format) (S, S, 25B)
		"""
		choose_truth_index = [None for _ in range(pred.shape[0])]
		iou = [0 for _ in range(pred.shape[0])]

		for i in range(pred.shape[0]):
			score_hat, cat_hat = pred[i][5:25].max(dim=0)
			confidence_hat = pred[i][4]
			# filter by confidence threshold
			if confidence_hat * score_hat < self.confidence_thres: continue

			x1hat, y1hat, x2hat, y2hat = pred[i][0:4]

			for j in range(truth.shape[0]):
				score, cat = truth[j][5:25].max(dim=0)
				confidence = truth[j][4]
				# filter true truth
				if confidence < 1: continue
				# judge whether is same class
				if cat != cat_hat: continue
				# calculate IoU
				x1, y1, x2, y2 = truth[j][0:4]

				wi = min(x2, x2hat) - max(x1, x1hat)
				hi = min(y2, y2hat) - max(y1, y1hat)
				wi = max(0, wi)
				hi = max(0, hi)
				intersection = wi * hi
				union = (x2 - x1) * (y2 - y1) + (x2hat - x1hat) * (y2hat - y1hat) - intersection
				this_iou = intersection / (union + 1e-6)
				# determine whether to choose this ground truth
				if iou[i] is None: choose = True
				elif iou[i] < this_iou: choose = True
				else: choose = False
				# if choose, assign value
				if choose:
					iou[i] = float(this_iou)
					choose_truth_index[i] = j
		# init a bool array for judging mustbe_FP later
		truth_chosen = [False for _ in range(len(truth))]
		# sort according to IoU
		sort_idx = np.argsort(iou)[::-1]
		# add into metrics
		for i in sort_idx:
			score, cat = pred[i][5:25].max(dim=0)
			confidence = pred[i][4]
			# filter by confidence threshold
			if confidence * score < self.confidence_thres: continue

			truth_index = choose_truth_index[i]
			if truth_index == None: 
				mustbe_FP = True
			elif truth_chosen[truth_index]:
				mustbe_FP = True
			else: 
				mustbe_FP = False
				truth_chosen[choose_truth_index[i]] = True
			
			self.data[cat]['data'].append(CalculationMetrics(iou[i], float(confidence * score), mustbe_FP))

			# update detection statistics
			self.data[cat]['detection'] += 1
		# update ground truth statistics
		for i in range(truth.shape[0]):
			score, cat = truth[i][5:25].max(dim=0)
			confidence = truth[i][4]
			if confidence < 1: continue
			self.data[cat]['truth'] += 1


	def add_data(self, pred: torch.Tensor, truth: torch.Tensor):
		"""Add data for calculating metrics

		Args:
			pred (torch.Tensor): detection prediction, can be either batch result or single result (#, S, S, 25B) or (S, S, 25B)
			truth (torch.Tensor): ground truth (YOLO v2 format), can be either batch result or single result (#, S, S, 25B) or (S, S, 25B)
		"""
		converter = Yolo2BBox()
		
		pred = converter(pred)
		truth = converter(truth)

		# check shape
		if len(pred.shape) == 2:
			pred.unsqueeze_(0)
		if len(truth.shape) == 2:
			truth.unsqueeze_(0)
		# assert size
		assert pred.shape[0] == truth.shape[0]
		cnt = pred.shape[0]

		for i in range(cnt):
			self._add_data(pred[i], truth[i])


	def calculate_precision_recall(self, iou_thres: float, class_idx: int) -> list:
		"""Calculate Precision-Recall Data according to IoU threshold

		Args:
			iou_thres (float): IoU threshold
			class_idx (int): Class Index

		Returns:
			list: `[{"precision": <precision>, "recall": <recall>}]`
		"""
		ret = []
		# retrieve count
		truth_cnt = self.data[class_idx]['truth']
		# accumulated TP
		acc_TP = 0
		# sort metrics by confidence
		data = sorted(self.data[class_idx]['data'], key=cmp_to_key(compare_metrics))
		for i, metrics in enumerate(data):
			if metrics.IoU >= iou_thres and not metrics.mustbe_FP:
				acc_TP += 1
			ret.append({
				'precision': acc_TP / (i + 1),
				'recall': acc_TP / truth_cnt
			})
		
		return ret


	def calculate_average_precision(self, iou_thres: float, class_idx: int, itpl_option: InterpolationMethod) -> float:
		"""Calculate Average Precision (AP)

		Args:
			iou_thres (float): IoU Threshold
			class_idx (int): Class Index
			itpl_option (InterpolationMethod): Interpolation Method

		Returns:
			float: AP of specified class using provided interpolation method
		"""
		prl = self.calculate_precision_recall(iou_thres=iou_thres, class_idx=class_idx)

		if itpl_option == InterpolationMethod.Interpolation_11:
			intp_pts = [0.1 * i for i in range(11)]
		elif itpl_option == InterpolationMethod.Interpolation_101:
			intp_pts = [0.01 * i for i in range(101)]
		else:
			raise Exception('Unknown Interpolation Method')

		max_dict = {}
		gmax = 0

		for pr in prl[::-1]:
			gmax = max(gmax, pr['precision'])
			max_dict[pr['recall']] = gmax

		if len(max_dict) < 1: return 0.

		max_keys = max_dict.keys()
		max_keys = sorted(max_keys)

		key_ptr = len(max_keys) - 2
		last_key = max_keys[-1]

		AP = 0

		for query in intp_pts[::-1]:
			if key_ptr < 0:
				if query > last_key:
					ans = 0
				else:
					ans = max_dict[last_key]
			else:
				if query > last_key:
					ans = 0
				elif query > max_keys[key_ptr]:
					ans = max_dict[last_key]
				else:
					while key_ptr >= 0:
						if query > max_keys[key_ptr]:
							break
						last_key = max_keys[key_ptr]
						key_ptr -= 1
					ans = max_dict[last_key]
			AP += ans

		AP /= len(intp_pts)
		return AP


	def calculate_mAP(self, iou_thres: float, itpl_option: InterpolationMethod) -> float:
		"""calculate mAP using given IoU threshold and interpolation method

		Args:
			iou_thres (float): IoU threshold
			itpl_option (InterpolationMethod): Interpolation Method

		Returns:
			float: Mean Average Precision (mAP)
		"""
		mAP = 0
		for c in range(len(self.data)):
			mAP += self.calculate_average_precision(iou_thres, c, itpl_option)
		mAP /= len(self.data)

		return mAP


	def calculate_VOCmAP(self) -> float:
		"""calculate VOCmAP: mAP with IoU thres = .5, interpolate by 0.1

		Returns:
			float: VOC mAP
		"""
		return self.calculate_mAP(0.5, InterpolationMethod.Interpolation_11)


	def calculate_COCOmAP50(self) -> float:
		"""calculate COCO mAP @50 (AP@.5): expand VOCmAP50, interpolate by 0.01

		Returns:
			float: AP@.5
		"""
		return self.calculate_mAP(0.5, InterpolationMethod.Interpolation_101)


	def calculate_COCOmAP75(self) -> float:
		"""calculate COCO mAP @75 (AP@.75): AP@.5, but with IoU thres = .75

		Returns:
			float: AP@.75
		"""
		return self.calculate_mAP(0.75, InterpolationMethod.Interpolation_101)


	def calculate_COCOmAP(self) -> float:
		"""calculate COCO mAP: expand AP@.5 and AP@.75. IoU thres from .5 to .95

		Returns:
			float: COCO mAP
		"""
		ious = [0.5 + 0.05 * i for i in range(10)]
		coco_map = 0
		for iou in ious:
			coco_map += self.calculate_mAP(iou, InterpolationMethod.Interpolation_101)
		coco_map /= len(ious)
		return coco_map
