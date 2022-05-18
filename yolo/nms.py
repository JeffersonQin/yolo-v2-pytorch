import torch
import torch.nn as nn
import torchvision
from utils import G


class YoloNMS(nn.Module):
	"""apply non-max suppression to the Yolo2BBox Result"""
	def __init__(self, iou_threshold: float = 0.5):
		"""apply non-max suppression to the Yolo2BBox Result

		Args:
			iou_threshold (float): IoU threshold for non-max suppression
		"""
		super(YoloNMS, self).__init__()
		self.iou_threshold = iou_threshold


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""forward

		Args:
			X (torch.Tensor): Single batch Yolo2BBox output, (S*S*B, (5+num_classes)) ((5+num_classes): x1, y1, x2, y2, objectness, class_prob)

		Returns:
			torch.Tensor: (count_after_nms, (5+num_classes)) ((5+num_classes): x1, y1, x2, y2, objectness, class_prob)
		"""
		with torch.no_grad():
			num_classes = G.get('num_classes')

			score, cat = X[:, 5:(5 + num_classes)].max(dim=1)
			nms_idx = torchvision.ops.batched_nms(X[:, 0:4], X[:, 4] * score, cat, self.iou_threshold)

			return X[nms_idx]
