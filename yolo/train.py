import os
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

from torch.utils.data import DataLoader
from utils import metrics as metrics_utils
from utils.utils import Accumulator, Timer, get_all_gpu, update_lr
from utils import G
from yolo.loss import YoloLoss


def train(net: nn.Module, train_iter: DataLoader, test_iter: DataLoader, num_epochs: int, lr, momentum: float, weight_decay: float, log_id: str, num_gpu: int=1, accum_batch_num: int=1, save_dir: str='./model', load: Optional[str]=None, load_epoch: int=-1, visualize_cnt: int=10):
	"""trainer for yolo v2. 
	Note: weight init is not done in this method, because the architecture
	of yolo v2 is rather complicated with the design of pass through layer

	Args:
		net (nn.Module): module network
		train_iter (DataLoader): training dataset iterator
		test_iter (DataLoader): testing dataset iterator
		num_epochs (int): number of epochs to train
		lr (float | callable): learning rate or learning rate scheduler function relative to epoch
		momentum (float): momentum for optimizer
		weight_decay (float): weight decay for optimizer
		log_id (str): identifier for logging in tensorboard.
		num_gpu (int, optional): number of gpu to train on, used for parallel training. Defaults to 1.
		accum_batch_num (int, optional): number of batch to accumulate gradient, used to solve OOM problem when using big batch sizes. Defaults to 1.
		save_dir (str, optional): saving directory for model weights. Defaults to './model'.
		load (Optional[str], optional): path of model weights to load if exist. Defaults to None.
		load_epoch (int, optional): done epoch count minus one when loading, should be the same with the number in auto-saved file name. Defaults to -1.
		visualize_cnt (int, optional): number of batches to visualize each epoch during training progress. Defaults to 10.
	"""
	os.makedirs(save_dir, exist_ok=True)

	# tensorboard
	writer = tensorboard.SummaryWriter(f'logs/yolo')

	# set up loading
	if load:
		net.load_state_dict(torch.load(load))

	# set up devices
	if not torch.cuda.is_available():
		net = net.to(torch.device('cpu'))
		devices = [torch.device('cpu')]
	else:
		if num_gpu > 1:
			net = nn.DataParallel(net, get_all_gpu(num_gpu))
			devices = get_all_gpu(num_gpu)
		else:
			net = net.to(torch.device('cuda'))
			devices = [torch.device('cuda')]

	# set up optimizer
	if isinstance(lr, float):
		tlr = lr
	else: tlr = 0.001
	optimizer = torch.optim.SGD(net.parameters(), tlr, momentum=momentum, weight_decay=weight_decay)

	# set up loss
	loss = YoloLoss()

	num_batches = len(train_iter)

	def plot(batch: int, num_batches: int, visualize_cnt: int) -> int:
		"""judge whether to plot or not for a specific batch

		Args:
			batch (int): batch count (starts from 1)
			num_batches (int): total batch count
			visualize_cnt (int): how many batches to visualize for each epoch

		Returns:
			int: if plot, return plot indicies (1 ~ visualize_cnt), else return 0
		"""
		if num_batches % visualize_cnt == 0:
			if batch % (num_batches // visualize_cnt) == 0:
				return batch // (num_batches // visualize_cnt)
			else:
				return 0
		else:
			if batch % (num_batches // visualize_cnt) == 0:
				if batch // (num_batches // visualize_cnt) == visualize_cnt:
					return 0
				else:
					return batch // (num_batches // visualize_cnt)
			elif batch == num_batches:
				return visualize_cnt
			else:
				return 0


	# train
	for epoch in range(num_epochs - load_epoch - 1):
		# adjust true epoch number according to pre_load
		epoch = epoch + load_epoch + 1

		# define metrics: train loss, sample count
		metrics = Accumulator(2)
		# define timer
		timer = Timer()

		# train
		net.train()

		# set batch accumulator
		accum_cnt = 0
		accum = 0

		# iterate over batches
		timer.start()
		for i, batch in enumerate(train_iter):

			X, y = batch
			X, y = X.to(devices[0]), y.to(devices[0])
			yhat = net(X)

			loss_val = loss(yhat, y, epoch)

			# backward to accumulate gradients
			loss_val.sum().backward()
			# update batch accumulator
			accum += 1
			accum_cnt += loss_val.shape[0]
			# step when accumulator is full
			if accum == accum_batch_num or i == num_batches - 1:
				# update learning rate per epoch and adjust by accumulated batch_size
				if callable(lr):
					update_lr(optimizer, lr(epoch) / accum_cnt)
				else:
					update_lr(optimizer, lr / accum_cnt)
				# step
				optimizer.step()
				# clear
				optimizer.zero_grad()
				accum_cnt = 0
				accum = 0

			with torch.no_grad():
				metrics.add(loss_val.sum(), X.shape[0])

			# log train loss
			print(f'epoch {epoch} batch {i + 1}/{num_batches} loss: {metrics[0] / metrics[1]}')
			plot_indices = plot(i + 1, num_batches, visualize_cnt)
			if plot_indices > 0:
				writer.add_scalars(f'loss/{log_id}', {
					'train': metrics[0] / metrics[1],
				}, epoch * visualize_cnt + plot_indices)

			# random choose a new image dimension size from
			# [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
			# that is, randomly adjust S between [10, 19]
			if epoch % 10 == 0:
				G.set('S', random.randint(10, 19))

		timer.stop()
		# log train timing
		writer.add_scalars(f'timing/{log_id}', {'train': timer.sum()}, epoch)

		# test!
		G.set('S', 13)
		G.set('B', 5)
		net.eval()
		metrics, timer = Accumulator(2), Timer()
		with torch.no_grad():
			timer.start()

			# test loss
			for i, batch in enumerate(test_iter):
				X, y = batch
				X, y = X.to(devices[0]), y.to(devices[0])
				yhat = net(X)

				loss_val = loss(yhat, y, 1000000) # very big epoch number to omit prior loss
				metrics.add(loss_val.sum(), X.shape[0])

				print(f'epoch {epoch} batch {i + 1}/{len(test_iter)} test loss: {metrics[0] / metrics[1]}')

			timer.stop()

			# log test loss
			writer.add_scalars(f'loss/{log_id}', {
				'test': metrics[0] / metrics[1],
			}, (epoch + 1) * visualize_cnt)
			# log test timing
			writer.add_scalars(f'timing/{log_id}', {'test': timer.sum()}, epoch)

			# test mAP every 5 epochs
			if (epoch + 1) % 5 == 0:
				calc = metrics_utils.ObjectDetectionMetricsCalculator(G.get('num_classes'), 0.1)

				for i, batch in enumerate(test_iter):
					X, y = batch
					X, y = X.to(devices[0]), y.to(devices[0])
					yhat = net(X)
					calc.add_data(yhat, y)

					print(f'epoch {epoch} batch {i + 1}/{len(test_iter)} testing mAP')

				# log test mAP
				writer.add_scalars(f'mAP/VOC', {log_id: calc.calculate_VOCmAP()}, epoch)
				writer.add_scalars(f'mAP/COCO', {log_id: calc.calculate_COCOmAP()}, epoch)
				writer.add_scalars(f'mAP/AP@.5', {log_id: calc.calculate_COCOmAP50()}, epoch)
				writer.add_scalars(f'mAP/AP@.75', {log_id: calc.calculate_COCOmAP75()}, epoch)

		# save model
		torch.save(net.state_dict(), os.path.join(save_dir, f'./{log_id}-epoch-{epoch}.pth'))
