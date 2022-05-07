import torch
import torch.nn as nn

from utils import globalvar as G
from yolo.converter import Yolo2BBox


class YoloLoss(nn.Module):
	"""Yolo loss."""
	def __init__(self, 
		lambda_coord: float = 1.0, 
		lambda_noobj: float = 1.0, 
		lambda_obj: float = 5.0, 
		lambda_class: float = 1.0, 
		lambda_prior: float = 0.01, 
		IoU_thres: float = 0.5, 
		epoch_prior: int = 20):
		"""Yolo loss.

			The loss function in this version has still not yet solved the problem of 
			multiple objects in one cell during training. There will be only one responsible 
			bbox in one cell where there is object, and the IoU will only be calculated 
			with one ground truth (no matter how many ground truths are there in one cell).

			Modified version of:
				https://www.cnblogs.com/YiXiaoZhou/p/7429481.html
			Loss = 
				# coordinate loss for responsible bbox
				# prior box loss (used to learn shape of prior boxes)
				#            => in other words, let t_w, t_h close to zero. (e^0 = 1)
				# class loss for all bboxes with obj (using only one ground truth)
				# objectness loss for bbox with best IoU less than IoU threshold
				# objectness loss for responsible bbox

			Args:
				yhat (torch.Tensor): yhat, [#, S, S, (5+num_classes)*B], where B is the number of bounding boxes.
				y (torch.Tensor): y, [#, S, S, (5+num_classes)*B], where B is the number of bounding boxes.

			Returns:
				torch.Tensor: loss [#]

		Args:
			lambda_coord (float, optional): lambda for coordinates. Defaults to 1.0.
			lambda_noobj (float, optional): lambda for no_obj, used for objectness. Defaults to 1.0.
			lambda_obj (float, optional): lambda for obj, used for objectness. Defaults to 5.0.
			lambda_class (float, optional): lambda for classes. Defaults to 1.0.
			lambda_prior (float, optional): lambda for prior boxes. Defaults to 0.01.
			IoU_thres (float, optional): IoU threshold while determining no_obj. Defaults to 0.5.
			epoch_prior (int, optional): epoch for learning prior boxes. Defaults to 20.
		"""
		super(YoloLoss, self).__init__()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.lambda_obj = lambda_obj
		self.lambda_class = lambda_class
		self.lambda_prior = lambda_prior
		self.IoU_thres = IoU_thres
		self.epoch_prior = epoch_prior


	def forward(self, yhat: torch.Tensor, y: torch.Tensor, epoch: int) -> list[torch.Tensor]:
		"""Calculate yolo loss.

		Args:
			yhat (torch.Tensor): yhat, [#, S, S, (num_classes+5)*B], where B is the number of bounding boxes.
			y (torch.Tensor): y, [#, S, S, (num_classes+5)*B], where B is the number of bounding boxes.
			epoch (int): epoch.

		Returns:
			list[torch.Tensor]: [#] coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss
		"""
		S = G.get('S')
		B = G.get('B')
		num_classes = G.get('num_classes')

		yhat = yhat.reshape(-1, S, S, B, 5 + num_classes)
		y = y.reshape(-1, S, S, B, 5 + num_classes)
		# pick first ground truth for each cell
		y = y[:, :, :, [0], :].expand(-1, S, S, B, 5 + num_classes)

		with torch.no_grad():
			# convert data into bbox format for IoU calculation
			converter = Yolo2BBox()
			yhat_bbox = converter(yhat).reshape(-1, S, S, B, 5 + num_classes)
			y_bbox = converter(y).reshape(-1, S, S, B, 5 + num_classes)
			
			# calculate IoU
			wi = torch.min(yhat_bbox[..., 2], y_bbox[..., 2]) - torch.max(yhat_bbox[..., 0], y_bbox[..., 0])
			wi = torch.max(wi, torch.zeros_like(wi))
			hi = torch.min(yhat_bbox[..., 3], y_bbox[..., 3]) - torch.max(yhat_bbox[..., 1], y_bbox[..., 1])
			hi = torch.max(hi, torch.zeros_like(hi))

			intersection = wi * hi
			union = (yhat_bbox[..., 2] - yhat_bbox[..., 0]) * (yhat_bbox[..., 3] - yhat_bbox[..., 1]) + \
				(y_bbox[..., 2] - y_bbox[..., 0]) * (y_bbox[..., 3] - y_bbox[..., 1]) - intersection
			IoU = intersection / (union + 1e-6)

			# filter out IoU < IoU_thres
			no_obj_iou = IoU < self.IoU_thres

			# pick responsible bbox indicies
			_, res = IoU.max(dim=3, keepdim=True)

		# pick responsible data
		yhat_res = torch.take_along_dim(yhat, res.unsqueeze_(3), 3).squeeze(3)
		y_res = y[:, :, :, 0, :]

		with torch.no_grad():
			# [#, S, S]
			have_obj = y_res[..., 4] > 0
			no_obj = ~have_obj

		# calculate loss
		# 1. coordinate loss
		coord_loss = ((yhat_res[:, :, :, 0:4] - y_res[:, :, :, 0:4]) ** 2).sum(dim=3) \
			* have_obj * self.lambda_coord * (2 - y_res[:, :, :, 2] * y_res[:, :, :, 3])
		coord_loss = coord_loss.sum(dim=(1, 2))
		# 2. class loss
		class_loss = ((yhat[:, :, :, :, 5:] - y[:, :, :, :, 5:]) ** 2).sum(dim=(3, 4)) \
			* have_obj * self.lambda_class
		class_loss = class_loss.sum(dim=(1, 2))
		# 3. no_obj loss
		no_obj_loss = ((yhat[:, :, :, :, 4] - y[:, :, :, :, 4]) ** 2) \
			* no_obj_iou * self.lambda_noobj
		no_obj_loss = no_obj_loss.sum(dim=(1, 2, 3))
		# 4. obj loss
		obj_loss = (yhat_res[:, :, :, 4] - y_res[:, :, :, 4]) ** 2 \
			* have_obj * self.lambda_obj
		obj_loss = obj_loss.sum(dim=(1, 2))
		# 5. prior loss
		if epoch < self.epoch_prior:
			anchors = G.get('anchors').to(yhat.device)
			prior_loss = ((yhat[:, :, :, :, 2:4] - anchors) ** 2).sum(dim=(3, 4)) \
				* no_obj * self.lambda_prior
			prior_loss = prior_loss.sum(dim=(1, 2))
		else: prior_loss = torch.Tensor([0]).to(yhat.device)

		return coord_loss, class_loss, no_obj_loss, obj_loss, prior_loss
