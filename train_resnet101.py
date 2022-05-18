import random
import numpy
import torch
from utils import G
from utils import data
from yolo.loss import YoloLoss
from yolo.model import ResNet101YoloDetector
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
multi_scale_epoch = 150
output_scale_S = 17
weight_decay = 0.0005
momentum = 0.9
lambda_coord = 1.0
lambda_noobj = 1.0
lambda_obj = 5.0
lambda_class = 1.0
lambda_prior = 0.01
IoU_thres = 0.6
epoch_prior = 20

# learning rate scheduler
def lr(epoch):
	if epoch < 10: return 0.00001 * (epoch + 1)
	if epoch < 20: return 0.0001 * (epoch - 9)
	if epoch < 60: return 0.001
	if epoch < 105: return 0.0001
	return 0.00001

loss = YoloLoss(lambda_coord, lambda_noobj, lambda_obj, lambda_class, lambda_prior, IoU_thres, epoch_prior)


if __name__ == '__main__':
	# data loader
	train_iter, test_iter = data.load_data_voc(batch_size, train_shuffle=True, test_shuffule=False, data_augmentation=True)

	# define network
	detector = ResNet101YoloDetector()

	# weight init
	detector.winit()

	optimizer = torch.optim.SGD(detector.parameters(), lr=lr(0), weight_decay=weight_decay, momentum=momentum)

	# train
	train(detector, train_iter, test_iter, num_epoch, multi_scale_epoch, output_scale_S, lr, optimizer, 'resnet101-sgd-17x17', loss, 1, accum_batch_num, './model', None, None, -1)
