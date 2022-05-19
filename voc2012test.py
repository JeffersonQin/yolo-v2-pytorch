import cv2
import os
import tarfile
from PIL import Image
import torch
import torchvision

from yolo.converter import Yolo2BBox
from yolo.model import ResNet101YoloDetector, ResNet18YoloDetector
from yolo.nms import YoloNMS
from utils import visualize
from utils import G


data_set_dir = '../data/VOC2012test'
result_base_dir = 'results'
result_file_name = 'results.tar.gz'

results_dir = os.path.join(result_base_dir, 'VOC2012/Main/')
test_index = os.path.join(data_set_dir, 'VOCdevkit/VOC2012/ImageSets/Main/test.txt')
image_path = os.path.join(data_set_dir, 'VOCdevkit/VOC2012/JPEGImages/')
categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

G.init(S=17, B=5)
os.makedirs(results_dir, exist_ok=True)
converter = Yolo2BBox()
nms = YoloNMS()


def test(net, comp_index, batch_size):
	lines = []
	with open(test_index, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			lines.append(line)

	with torch.no_grad():
		idx = 0
		while idx < len(lines):
			print(f'Batch: {idx} / {len(lines)}')
			to_idx = min(len(lines), idx + batch_size)
			batch_lines = lines[idx:to_idx]
			idx = to_idx

			X = torch.Tensor([])
			X = X.to('cuda')

			S = torch.Tensor([])
			S = S.to('cuda')

			for line in batch_lines:
				image_name = line + '.jpg'
				this_path = os.path.join(image_path, image_name)

				with Image.open(this_path) as img:
					s = torch.Tensor([float(img.size[0]), float(img.size[1])])
					s = s.to('cuda')
					S = torch.cat((S, s.unsqueeze_(0)), 0)

					rimg = torchvision.transforms.functional.resize(img, (G.get('S') * 32, G.get('S') * 32))
					t = torchvision.transforms.ToTensor()(rimg).to('cuda')
					X = torch.cat((X, t.unsqueeze_(0)), 0)

			YHat = net(X)
			YHat = converter(YHat)
			for i, yhat in enumerate(YHat):
				yhat = nms(yhat)
				# render
				# res = visualize.draw_detection_result(visualize.tensor_to_cv2(X[i]), yhat, thres=0.1)
				# cv2.imwrite(f'./results/{batch_lines[i]}.jpg', res)

				W = S[i][0]
				H = S[i][1]

				category_detected = [False for _ in range(20)]

				for j in range(yhat.shape[0]):
					x1, y1, x2, y2, iou = yhat[j][0:5]

					score, cat = yhat[j][5:25].max(dim=0)
					if iou * score < 0.1: continue

					file_name = f'comp{comp_index}_det_test_{categories[cat]}.txt'
					file_path = os.path.join(results_dir, file_name)

					category_detected[cat] = True

					with open(file_path, 'a+', encoding='utf-8', newline='\n') as f:
						x1 = max(1, int(x1 * W))
						y1 = max(1, int(y1 * H))
						x2 = min(int(W), int(x2 * W))
						y2 = min(int(H), int(y2 * H))
						conf = round(float(score * iou), 6)
						f.write(f'{batch_lines[i]} {conf} {x1}.000000 {y1}.000000 {x2}.000000 {y2}.000000\n')


if __name__ == '__main__':

	def test_resnet18():
		# load model
		model_weight_path = './model/resnet18-pretrained-sgd-epoch-160.pth'
		net = ResNet18YoloDetector()
		net.to('cuda')

		net.load_state_dict(torch.load(model_weight_path))
		net.eval()

		G.set('S', 17)
		test(net, 3, 64)

	def test_resnet101():
		# load model
		model_weight_path = './model/resnet101-pretrained-sgd-17x17-epoch-160.pth'
		net = ResNet101YoloDetector()
		net.to('cuda')

		net.load_state_dict(torch.load(model_weight_path))
		net.eval()

		G.set('S', 17)
		test(net, 4, 64)

	# test_resnet18()
	# test_resnet101()

	with tarfile.open(result_file_name, 'w:gz') as tar:
		tar.add(results_dir)
