import cv2
import torch
import torchvision
from utils import G, visualize
from utils.utils import Timer
from yolo.converter import Yolo2BBox
from yolo.model import ResNet101YoloDetector
from yolo.nms import YoloNMS

S = 17
A = S * 32

G.init()
G.set('S', S)

detector = ResNet101YoloDetector()
detector.load_state_dict(torch.load('./model/resnet101-pretrained-sgd-17x17-epoch-160.pth'))
detector.eval()
detector.to('cuda')

converter = Yolo2BBox()
nms = YoloNMS()

timer = Timer()
to_tensor = torchvision.transforms.ToTensor()

cap = cv2.VideoCapture(0)

while True:
	timer.start()
	ret, frame = cap.read()
	if not ret: break

	frame = cv2.resize(frame, (A, A), interpolation=cv2.INTER_LINEAR)
	X = to_tensor(frame).to('cuda').flip(0).unsqueeze_(0)

	pred = nms(converter(detector(X))[0])
	cv2.imshow('Webcam', visualize.draw_detection_result(frame, pred, thres=0.1))
	timer.stop()

	print(timer.avg())
	cv2.waitKey(1)
