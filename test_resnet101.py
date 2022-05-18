import torch
from utils import G, metrics
from utils import data
from utils.utils import Timer
from yolo.converter import Yolo2BBox
from yolo.model import ResNet101YoloDetector
from yolo.nms import YoloNMS


G.init()
G.set('S', 17)

detector = ResNet101YoloDetector()
detector.load_state_dict(torch.load('./model/resnet101-pretrained-sgd-17x17-epoch-160.pth'))
detector.eval()
detector.to('cuda')

converter = Yolo2BBox()
nms = YoloNMS()

if __name__ == '__main__':
	with torch.no_grad():
		# test FPS
		_, test_iter = data.load_data_voc(batch_size=1, train_shuffle=False, test_shuffule=False, data_augmentation=False)
		timer = Timer()
		for i, (X, _) in enumerate(test_iter):
			print(f'FPS Test, Image No. {i} / {len(test_iter)}')

			timer.start()
			X = X.to('cuda')
			Y = detector(X)
			Y = converter(Y)
			nms(Y[0])
			timer.stop()

		print('FPS: %.2f' % (1 / timer.avg()))

		# test mAP
		_, test_iter = data.load_data_voc(batch_size=32, train_shuffle=False, test_shuffule=False, data_augmentation=False)
		calc = metrics.ObjectDetectionMetricsCalculator(G.get('num_classes'), 0.1)
		for i, (X, Y) in enumerate(test_iter):
			print(f'mAP Test, Batch {i} / {len(test_iter)}')

			X = X.to('cuda')
			Y = Y.to('cuda')
			YHat = detector(X)
			calc.add_data(YHat, Y)

		print(f'FPS: {1 / timer.avg()}')
		print(f'VOCmAP: {calc.calculate_VOCmAP()}')
		print(f'COCOmAP: {calc.calculate_COCOmAP()}')
