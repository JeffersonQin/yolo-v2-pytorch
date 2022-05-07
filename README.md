# YOLO v2 PyTorch Implementation

**I wrote this repo for the purpose of learning, aimed to reproduce YOLO v2 using PyTorch.** Most of the **ideas** were adopted from the original paper, but ... it is **extremely difficult to decrypt the mysterious code of Darknet**, so ... I crafted my own version on some designs.

As normal, I do not have the condition to pretrain myself, so I modified the architecture and mainly focused on the implementation of ideas and loss of YOLO.

Also, hyperparameters are ... *randomly* chosen by me depended on my mood 'cause I no longer want to spend effort on the catastrophic source code ...

Besides, compared with the [YOLO v1 I implemented](https://github.com/JeffersonQin/yolo-v1-pytorch), I believed this version is pretty nice and the code is far more organized, and the efficiency of mAP calculation is improved tremendously, I've made it possible to calculate mAP each epoch with considerable time cost. Here comes the project structure.

```
.
├── kmeans.py             # script to calculate prior box on VOC train dataset
├── train_yolov2.py       # train yolo v2 Darknet19 (w/o pretrain)
├── utils                 # utils
│   ├── __init__.py
│   ├── data.py           # data pipeline, augmentation
│   ├── globalvar.py      # global variable
│   ├── metrics.py        # mAP calculation
│   ├── utils.py          # utils
│   ├── visualize.py      # visualization
│   └── winit.py          # weight init
└── yolo                  # YOLO model related
    ├── __init__.py
    ├── converter.py      # data converter, BBox <=> model output
    ├── loss.py           # YOLO loss module
    ├── model.py          # network
    ├── nms.py            # non-maximum suppression
    └── train.py          # trainer
```

## YOLO v2 Features

* ✅: Implemented and used
* ❌: Not implemented
* \*: Not available for other backbones

|       Tricks        | Used  |
| :-----------------: | :---: |
|      BatchNorm      |   ✅   |
|  Hi-Res Classifier  |   ❌   |
|    Convolutional    |   ✅   |
|     Anchor Box      |   ✅   |
|     New Network     |  ✅*   |
|   Dimension Prior   |   ✅   |
| Location Prediction |   ✅   |
|     Passthrough     |  ✅*   |
|     Multi-scale     |   ✅   |
|   Hi-Res Detector   |   ✅   |

## YOLO v2 Loss

Loss function of YOLO v2 was not given explicitly from the paper. I've tried my best to read the source code of Darknet...

Here are also some references:

* [YOLO v2 损失函数源码分析](https://www.cnblogs.com/YiXiaoZhou/p/7429481.html)
* [Training Object Detection (YOLOv2) from scratch using Cyclic Learning Rates](https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755)

And here is my version of loss:

<!-- $$
	\begin{aligned}
		&\lambda_{\text{coord}} \sum_{i=1}^{S^2}\sum_{j=1}^B 1_{ij}^{\text{obj}} (2-w_i\times h_i)[(x_i-\hat x_{ij})^2 + (y_i-\hat y_{ij})^2 + (w_i-\hat w_{ij})^2 + (h_i-\hat h_{ij})^2] \\
		+&\lambda_{\text{class}} \sum_{i=1}^{S^2}\sum_{j=1}^B 1_{i}^{\text{obj}} \sum_{c\in \text{classes}} (p_i(c) - \hat p_{ij}(c))^2 \\ 
		+&\lambda_{\text{noobj}} \sum_{i=1}^{S^2}\sum_{j=1}^B 1_{\text{MaxIoU}_{ij} < \text{IoUThres}}(C_{ij} - \hat C_{ij})^2 \\ 
		+&\lambda_{\text{obj}} \sum_{i=1}^{S^2}\sum_{j=1}^B 1_{ij}^{\text {obj}} (C_{ij} - \hat C_{ij})^2 \\
		+&\lambda_{\text{prior}} \sum_{i=1}^{S^2}\sum_{j=1}^B 1_{i}^{\text{noobj}} 1_{\text{iter} < 12800} [(\hat w_{ij} - p(w)_{j})^2 + (\hat h_{ij} - p(h)_{j})^2] 
	\end{aligned}
$$ --> 

<div align="center"><img style="background: white;" src="./assets/tN3mGO82jf.svg"></div>

## About Dimension Prior

<div align="center">
	<img src="./assets/dimension-cluster.svg" width=300>
</div>

It is known that in YOLO v2 and YOLO v3, anchor boxes were used. Here we implemented the k-means algorithm for finding dimension priors in `kmeans.py`, and the graph above are 5 priors we obtained by identifying k = 5.

If you want to run it by yourself, simply try

```
python kmeans.py
```
