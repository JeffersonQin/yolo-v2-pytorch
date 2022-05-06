import torch
import torch.nn as nn
from utils import globalvar as G
from yolo.converter import YoloAnchorLayer


class ConvUnit(nn.Module):
	"""Convolutional Unit, consists of conv2d, batchnorm, leaky_relu"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(ConvUnit, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
	
	def forward(self, X):
		X = self.conv(X)
		X = self.bn(X)
		X = self.leaky_relu(X)
		return X


class Darknet19Backbone(nn.Module):
	def __init__(self):
		super(Darknet19Backbone, self).__init__()
		self.conv1 = ConvUnit(3, 32, 3, 1, 1)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = ConvUnit(32, 64, 3, 1, 1)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = ConvUnit(64, 128, 3, 1, 1)
		self.conv4 = ConvUnit(128, 64, 1, 1, 0)
		self.conv5 = ConvUnit(64, 128, 3, 1, 1)
		self.pool3 = nn.MaxPool2d(2, 2)
		self.conv6 = ConvUnit(128, 256, 3, 1, 1)
		self.conv7 = ConvUnit(256, 128, 1, 1, 0)
		self.conv8 = ConvUnit(128, 256, 3, 1, 1)
		self.pool4 = nn.MaxPool2d(2, 2)
		self.conv9 = ConvUnit(256, 512, 3, 1, 1)
		self.conv10 = ConvUnit(512, 256, 1, 1, 0)
		self.conv11 = ConvUnit(256, 512, 3, 1, 1)
		self.conv12 = ConvUnit(512, 256, 1, 1, 0)
		self.conv13 = ConvUnit(256, 512, 3, 1, 1) # pass through
		self.pool5 = nn.MaxPool2d(2, 2)
		self.conv14 = ConvUnit(512, 1024, 3, 1, 1)
		self.conv15 = ConvUnit(1024, 512, 1, 1, 0)
		self.conv16 = ConvUnit(512, 1024, 3, 1, 1)
		self.conv17 = ConvUnit(1024, 512, 1, 1, 0)
		self.conv18 = ConvUnit(512, 1024, 3, 1, 1)

		self.head = nn.Sequential(
			self.conv1, self.pool1,
			self.conv2, self.pool2,
			self.conv3, self.conv4, self.conv5, self.pool3,
			self.conv6, self.conv7, self.conv8, self.pool4,
			self.conv9, self.conv10, self.conv11, self.conv12, self.conv13)

		self.tail = nn.Sequential(
			self.pool5,
			self.conv14, self.conv15, self.conv16, self.conv17, self.conv18)


	def forward(self, X):
		self.pass_through = self.head(X)
		return self.tail(self.pass_through)


class Darknet19Detection(nn.Module):
	def __init__(self, backbone: Darknet19Backbone):
		super(Darknet19Detection, self).__init__()
		self.backbone = backbone
		self.conv19 = ConvUnit(1024, 1024, 3, 1, 1)
		self.conv20 = ConvUnit(1024, 1024, 3, 1, 1)

		self.conv21 = ConvUnit(1024+256, 1024, 3, 1, 1)
		self.conv22 = ConvUnit(1024, (5 + G.get('num_classes')) * G.get('B'), 1, 1, 0)

		self.nin_conv = ConvUnit(512, 64, 1, 1, 0)
		self.anchor_layer = YoloAnchorLayer()

		self.head = nn.Sequential(self.backbone, self.conv19, self.conv20)
		self.tail = nn.Sequential(self.conv21, self.conv22, self.anchor_layer)


	def forward(self, X):
		X = self.head(X)

		ksize = self.backbone.pass_through.shape[2]
		assert ksize % 2 == 0
		ksize //= 2

		reshape = nn.Sequential(nn.Flatten(), nn.Unflatten(1, (256, ksize, ksize)))
		pass_through = reshape(self.nin_conv(self.backbone.pass_through))

		X = torch.cat((X, pass_through), dim=1)
		return self.tail(X)
