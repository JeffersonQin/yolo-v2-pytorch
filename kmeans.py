import json
import os
import torch
import torch.utils.tensorboard as tensorboard

from utils.data import load_data_voc
from utils.utils import Accumulator, try_gpu
from utils import globalvar as G


def train(data, batch_size, K, num_epoch, log_id='', device=try_gpu()):
	# init tensorboard
	writer = tensorboard.SummaryWriter(f'logs/kmeans-anchor{"" if log_id == "" else f"-{log_id}"}')
	# init cluster centroids
	mu = torch.rand((K, 2))
	mu = mu.to(device)

	for epoch in range(num_epoch):
		# iou, sample count
		metrics = Accumulator(2)

		WS = torch.zeros(K, device=device)
		HS = torch.zeros(K, device=device)
		WC = torch.zeros(K, device=device)
		HC = torch.zeros(K, device=device)

		for Y in make_data_iter(data, batch_size):
			Y = Y.to(device)
			# [N]
			W = Y[:, 0]
			H = Y[:, 1]
			N = W.shape[0]
			# [K]
			WHat = mu[:, 0]
			HHat = mu[:, 1]
			# [N] => [N, 1] => [N, K]
			W = W.unsqueeze(1).expand(N, K)
			H = H.unsqueeze(1).expand(N, K)
			# [K] => [1, K] => [N, K]
			WHat = WHat.unsqueeze(0).expand(N, K)
			HHat = HHat.unsqueeze(0).expand(N, K)

			# calculate distance
			intersection = torch.min(WHat, W) * torch.min(HHat, H)
			union = W * H + WHat * HHat - intersection
			IoU = intersection / (union + 1e-6)
			D = 1 - IoU

			Dval, _ = D.min(dim=1, keepdim=True)
			# [N, K] => [K, N]
			W = ((Dval == D) * W).transpose(0, 1)
			H = ((Dval == D) * H).transpose(0, 1)

			# update centroids
			WS = WS + torch.sum(W, dim=1)
			HS = HS + torch.sum(H, dim=1)
			WC = WC + torch.count_nonzero(W, dim=1)
			HC = HC + torch.count_nonzero(H, dim=1)

			IoUVal, _ = IoU.max(dim=1, keepdim=True)
			# update metrics
			metrics.add(((IoUVal == IoU) * IoU).sum(), N)

		# update centroids
		mu[:, 0] = WS / (WC + 1e-6)
		mu[:, 1] = HS / (HC + 1e-6)

		# tensorboard
		writer.add_scalar(f'IoU/{K}', metrics[0] / metrics[1], epoch)
		print(f'K = {K}, Epoch {epoch + 1}: IoU = {metrics[0] / metrics[1]}')

	# save
	ret = []
	for i in range(K):
		ret.append({
			'width': float(mu[i][0]),
			'height': float(mu[i][1]),
		})
	
	os.makedirs('./anchors', exist_ok=True)
	with open(f'anchors/anchors-{k}.json', 'w') as f:
		json.dump(ret, f)

	writer.add_scalar(f'IoU/K', metrics[0] / metrics[1], K)


def load_data() -> torch.Tensor:
	"""load everything to memory

	Returns:
		torch.Tensor: [#, 2] (W, H)
	"""
	# load dataset
	train_iter, _ = load_data_voc(1024, train_shuffle=False, data_augmentation=False)

	# define ret
	Y = torch.Tensor()
	Y = Y.to(try_gpu())

	for i, (_, y) in enumerate(train_iter):
		y = y.to(try_gpu())
		y = y.reshape(-1, 25)
		p = (y[:, 4] == 1)
		Y = torch.cat((Y, y[:, 2:4][p]), dim=0)
		print(f'Loading Data: {i + 1} / {len(train_iter)}')
	return Y.to('cpu')


def make_data_iter(data: torch.Tensor, batch_size: int) -> torch.Tensor:
	"""make data iterator

	Args:
		data (torch.Tensor): data in memory
		batch_size (int): batch size

	Returns:
		torch.Tensor: batch data
	"""
	num = data.shape[0]
	for i in range(0, num, batch_size):
		yield data[i:min(i + batch_size, num)]


if __name__ == '__main__':
	G.init()

	data = load_data()

	for k in range(1, 16):
		train(data, batch_size=2048, K=k, num_epoch=200)
