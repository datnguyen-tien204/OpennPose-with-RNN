![header](github/image/daaf2d95fb02505c0913.jpg)
[English](README.md) | [简体中文](cn_readme.md) | [Tiếng việt](README.vietnam-vn.md)

## Contents
1. [OpenPose with Recurrent Neural Network](#introduction)
2. [Results](#results)
3. [Installation](#installation)
4. [Quick Start Overview](#quick-start-overview)
5. [Structures](#structures)
6. [Send Us Feedback!](#send-us-feedback)
7. [Thanks](#thanks)
8. [License](#license)


# Introduction

此项目提供了OpenPose + RNN的异常检测实现。为了简单起见，我们在此自述文件的其余部分中将此模型称为OpenPoseRNN。我们还要感谢Minh Chuan-Pham博士和Quoc Viet-Hoang博士对此项目的支持。这种基于深度学习的系统正在全球发达国家如英国、法国、美国以及日本、韩国、中国等亚洲国家中应用。一些大学如清华大学、北京大学、斯坦福大学等使用技术防止考试作弊。它正在与考试监考员合作实施，以达到最高的效果并确保考试的公平性。

# Results
### 作弊识别总结（使用OpenPose + Yolov3+ 循环神经网络）
<p align="center">
    <img src="github/video/video_demo.gif" width="1000">
    <br>
    <sup>使用12422TN类进行测试 <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose" target="_blank"><i> on OpenPose </i></a>
</p>

### For Human Detection
对于这部分，我们使用 [YOLOv3](https://github.com/ultralytics/yolov3) 用于检测房间中的人。为了评估这个模型，我们使用从原始数据集分割出来的trainval35k集。 [MS-COCO 2017](https://cocodataset.org/#home)  数据集. 结果显示在表1中。

**表 1. 与其他3种模型的人体检测图像的比较结果**
|    Models   | Avg. | Precision |  IoU |
|:-----------:|:----:|:---------:|:----:|
| [Faster-RCNN](https://arxiv.org/abs/1506.01497) | 21.9 |    42.7   |   -  |
|    SSD300   | 25.2 |    43.1   | 26.1 |
|    YOLOv2   | 21.6 |     44    | 19.2 |
|   **Ours**    |**25.3**| **44.5**   | **25.9** |

### 用于骨架位置定位

用于骨架位置定位, 我们使用 [OpenPose]("https://github.com/CMU-Perceptual-Computing-Lab/openpose) 为了检测人体骨骼，我们使用了这个模型进行评估 [MS-COCO2015](https://cocodataset.org/#home) 数据集。结果显示在表2中。

**表 2. 与其他两种模型相比，骨架位置定位的评估结果**
|     Models    |  AP@0.5  |  AP@0.75 | AP medium | Ap large |
|:-------------:|:--------:|:--------:|:---------:|----------|
| [AlphaPose]() |   89.2   |   79.1   |     69    | 78.6     |
|     Detectron Mask-RCNN    |   25.2   |   43.1   |    26.1   | 68.2     |
|    **Ours**   | **88.0** | **73.1** |  **62.2** | **78.6** |

此外，我们还使用 FPS 和 GPU 内存来评估这一点。结果显示在表 3 中，用于多人和表 4 中的单人.

**表 3. 多人结果**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   73.4%   |   1.15   |
|    **Ours**   | **21.3%** | **18.39** |

**表 4. 单人结果**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   60.3%   |   23.71   |
|    **Ours**   | **21.3%** | **18.77** |

### For recognition 
我们用了 [Recurrrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) 我们使用循环神经网络(RNN)来对考场中的行为进行分类。为了评估这部分，我们使用了两个指标，即混淆矩阵和接收者操作特征曲线(ROC)。结果如下所示 [图 1](github/image/confMatrix.jpg) and [图 1](github/image/RocCurves)

<p align="center">
    <img src="github/image/confMatrix.jpg" width="1000">
    <br>
    <sup>图 1. 所有标签的结果<a href="https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank"><i> Confusion Matrix  </i></a>
</p>

<p align="center">
    <img src="github/image/RocCurves.png" width="1000">
    <br>
    <sup>图 2. 所有标签的结果 <a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc" target="_blank"><i> ROC  </i></a>
</p>

        
# Installation

### With Python Base
Requirements python >= 3.7
1. Install dependences library
 ```bash
 pip install -r requirements.txt
```
2. Install dependences files
- Change directory to ``` OpenPose/graph_models/VGG_origin```, you can change directory with this command ```cd OpenPose/graph_models/VGG_origin ```
- After you must run ``` file_requirements.py ``` or
 ``` bash
python file_requirements.py
```
3. Install dependences files with other steps ( Optional )
- If you step 2 not successfully you can download weights from [Google Drive](https://drive.google.com/drive/folders/1Y4coXLsVzCXYuCKpyDfQBqpHH8Aj-Yg5?usp=sharing)
- Move folder ```graph_models``` downloaded to ```OpenPose\graph_models``` 

### With Anaconda 
1. Install dependences library
   - You can load dependences library with ``` openpose.yaml``` file.
   - You can find ```openpose.yaml``` file in folder ```Environment```
2. Install dependences files
- Change directory to ``` OpenPose/graph_models/VGG_origin```, you can change directory with this command ```cd OpenPose/graph_models/VGG_origin ```
- After you must run ``` file_requirements.py ``` or
 ``` bash
python file_requirements.py
```
3. Install dependences files with other steps ( Optional )
- If you step 2 not successfully you can download weights from [Google Drive](https://drive.google.com/drive/folders/1Y4coXLsVzCXYuCKpyDfQBqpHH8Aj-Yg5?usp=sharing)
- Move folder ```graph_models``` downloaded to ```OpenPose\graph_models``


# Quick Start Overview
### With Python Base Environments and Anaconda Environment
1. Quick Run
 - You can run this file ```main.py``` to start this project. 
2. [Optinal]To trainning model you using ```create_data.py``` to export data points and move to folder ```Action\trainning``` and using .ipnb file ```train.ipnb``` to train.
3. [Optinal] Using VGG_origin can be slow, if you don't have GPU you can change model to ```mobilenet``` to predict faster.
   - To change model to ```mobilenet```, navigation to file ```main.py``` in main folder.
   - In line 14, change ``` estimator = load_pretrain_model('VGG_origin')``` to ```estimator = 
 load_pretrain_model('mobilenet_thin')```
   
4.[Optinal] To use your weight, you can change it in ```main.py```, in line 15 change ```action_classifier = load_action_premodel('open_pose2\Action\framewise_recognition_under_scene.h5')``` to ```action_classifier = load_action_premodel('path_to_your_weights')```
  
# Structures
**Structures for all models**
<p align="center">
    <img src="github/image/structure.png" width="500">
    <br>
</p>

# Send Us FeedBack
Our project is open source for research purposes, and we want to improve it! So let us know (create a new GitHub issue or pull request, email us, etc.) if you...
1. Find/fix any bug (in functionality or speed) or know how to speed up or improve any part of OpenPoseRNN.
2. Want to add/show some cool functionality/demo/project made on top of Students Tracking. We can add your project link to your [Issue](https://github.com/datnguyen-tien204/Tracking_Students/issues)

# Thanks
Thank you for the guidance of Dr.Minh Chuan-Pham in the process of creating this project, as well as the evaluation board consisting of Dr.Quoc Viet-Hoang, who helped us improve the results and provided feedback for this project.

# License
This project is freely available for free non-commercial use. If it useful you can give 1 star. Thanks for using.
