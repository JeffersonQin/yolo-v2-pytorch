import torch
import torch.nn as nn
from utils import globalvar as G


__all__ = ['YoloAnchorLayer', 'Yolo2BBox']


class YoloAnchorLayer(nn.Module):
	"""Apply anchors to the output"""
	def __init__(self):
		"""Apply anchors to the output"""
		super(YoloAnchorLayer, self).__init__()
		self.anchors = G.get('anchors')


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		self.anchors = self.anchors.to(X.device)
		shape = X.shape
		S = G.get('S')
		B = G.get('B')
		num_classes = G.get('num_classes')

		# reshape from conv2d shape to (batch_size, S, S, filter)
		X = X.permute(0, 2, 3, 1)
		shape = X.shape

		# reshape to (batch_size, S, S, B, 5 + num_classes) for further processing
		X = X.reshape(-1, S, S, B, 5 + num_classes)

		XC = torch.clone(X)

		XC[..., 0:2] = X[..., 0:2].sigmoid()
		XC[..., 4:(5 + num_classes)] = X[..., 4:(5 + num_classes)].sigmoid()
		XC[..., 2] = X[..., 2].exp() * self.anchors[:, 0]
		XC[..., 3] = X[..., 3].exp() * self.anchors[:, 1]
		
		# reshape back
		XC = XC.reshape(shape)
		return XC


class Yolo2BBox(nn.Module):
	"""convert yolo result from (S, S, (5+num_classes)*B) or (#, S, S, (5+num_classes)*B) to normal bounding box result with size (S*S*B, (5+num_classes)) or (#, S*S*B, (5+num_classes))"""
	def __init__(self):
		"""convert yolo result from (S, S, (5+num_classes)*B) or (#, S, S, (5+num_classes)*B) to normal bounding box result with size (S*S*B, (5+num_classes)) or (#, S*S*B, (5+num_classes))"""
		super(Yolo2BBox, self).__init__()


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""forward

		Args:
			X (torch.Tensor): yolo result (S, S, (5+num_classes)*B) or (#, S, S, (5+num_classes)*B)

		Returns:
			torch.Tensor: bounding box result (S*S*B, (5+num_classes)) or (#, S*S*B, (5+num_classes)) ((5+num_classes): x1, y1, x2, y2, objectness, class_prob)
		"""
		with torch.no_grad():
			device = X.device
			S = G.get('S')
			B = G.get('B')
			num_classes = G.get('num_classes')

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

			single = False
			if len(X.shape) == 3:
				X.unsqueeze_(0)
				single = True

			X = X.reshape(-1, S, S, B, 5 + num_classes)
			x = (X[..., 0] + cell_xidx) / S
			y = (X[..., 1] + cell_yidx) / S

			x1 = x - X[..., 2] / 2.0
			y1 = y - X[..., 3] / 2.0
			x2 = x + X[..., 2] / 2.0
			y2 = y + X[..., 3] / 2.0

			XC = X.clone()

			XC[..., 0] = x1
			XC[..., 1] = y1
			XC[..., 2] = x2
			XC[..., 3] = y2

			XC = XC.reshape(-1, S * S * B, 5 + num_classes)

			if single:
				XC = XC[0]
			
			return XC
