import torch
import torch.nn as nn
from utils import globalvar as G


__all__ = ['YoloAnchorLayer', 'Yolo2BBox']


class YoloAnchorLayer(nn.Module):
	"""Apply anchor to the output"""
	def __init__(self):
		"""Apply anchor to the output"""
		super(YoloAnchorLayer, self).__init__()
		self.anchor = G.get('anchor')


	def forawrd(self, X: torch.Tensor) -> torch.Tensor():
		self.anchor = self.anchor.to(X.device)
		shape = X.shape
		S = G.get('S')
		B = G.get('B')
		X = X.reshape(-1, S, S, B, 25)
		X[..., 0:2].sigmoid_()
		X[..., 4:25].sigmoid_()
		X[..., 2] = X[..., 2].exp() * self.anchor[:, 0]
		X[..., 3] = X[..., 3].exp() * self.anchor[:, 1]
		X = X.reshape(shape)
		return X


class Yolo2BBox(nn.Module):
	"""convert yolo result from (S, S, 25B) or (#, S, S, 25B) to normal bounding box result with size (S*S*B, 25) or (#, S*S*B, 25)"""
	def __init__(self):
		"""convert yolo result from (S, S, 25B) or (#, S, S, 25B) to normal bounding box result with size (S*S*B, 25) or (#, S*S*B, 25)"""
		super(Yolo2BBox, self).__init__()


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""forward

		Args:
			X (torch.Tensor): yolo result (S, S, 25B) or (#, S, S, 25B)

		Returns:
			torch.Tensor: bounding box result (S*S*B, 25) or (#, S*S*B, 25) (25: x1, y1, x2, y2, objectness, class_prob)
		"""
		with torch.no_grad():
			device = X.device
			S = G.get('S')
			B = G.get('B')

			# arrange cell xidx, yidx
			# [S, S]
			cell_xidx = (torch.arange(S * S) % S).reshape(S, S)
			cell_yidx = (torch.div(torch.arange(S * S), S, rounding_mode='floor')).reshape(S, S)
			# transform to [S, S, B]
			cell_xidx.unsqueeze_(-1)
			cell_yidx.unsqueeze_(-1)
			cell_xidx.expand(S, S, B)
			cell_yidx.expand(S, S, B)
			# move to device
			cell_xidx = cell_xidx.to(device)
			cell_yidx = cell_yidx.to(device)

		with torch.no_grad():
			single = False
			if len(X.shape) == 3:
				X.unsqueeze_(0)
				single = True

			X.reshape_(-1, S, S, B, 25)
			x = (X[..., 0] + cell_xidx) / S
			y = (X[..., 1] + cell_yidx) / S

			x1 = x - X[..., 2] / 2.0
			y1 = y - X[..., 3] / 2.0
			x2 = x + X[..., 2] / 2.0
			y2 = y + X[..., 3] / 2.0

			X[..., 0] = x1
			X[..., 1] = y1
			X[..., 2] = x2
			X[..., 3] = y2

			X.reshape_(-1, S * S * B, 25)

			if single:
				X = X[0]
			
			return X
