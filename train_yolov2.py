import random
import numpy
import torch
from utils import G
from utils import data
from utils.winit import weight_init
from yolo.model import Darknet19Backbone, Darknet19Detection
from yolo.train import train

seed = 42
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

# init global variables
G.init()

# define hyper parameters
batch_size = 4
accum_batch_num = 16
num_epoch = 160
weight_decay = 0.0005
momentum = 0.9

# learning rate scheduler
def lr(epoch):
	if epoch < 10: return 0.0001 * (epoch + 1)
	if epoch < 60: return 0.001
	if epoch < 90: return 0.0001
	return 0.00001


if __name__ == '__main__':
	# data loader
	train_iter, test_iter = data.load_data_voc(batch_size, train_shuffle=True, test_shuffule=False, data_augmentation=True)

	# define network
	backbone = Darknet19Backbone()
	detector = Darknet19Detection(backbone)

	# weight init
	detector.apply(weight_init)

	# train
	train(detector, train_iter, test_iter, num_epoch, lr, momentum, weight_decay, 'darknet19-no-pretrain', 1, accum_batch_num, './model', None, -1)
