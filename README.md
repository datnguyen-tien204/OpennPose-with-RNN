![header](github/image/daaf2d95fb02505c0913.jpg)
[English](README.md) | [简体中文](README.zh-CN.md) | [Tiếng việt](README.vietnam-vn.md)

## Contents
1. [Introduction](#introduction)
2. [Results](#results)
3. [Installation](#installation)
4. [Quick Start Overview](#quick-start-overview)
5. [Structures](#structures)
6. [Send Us Feedback!](#send-us-feedback)
7. [Thanks](#thanks)
8. [License](#license)


# OpenPose with Recurrent Neural Network

This project provides an implementation anomaly detection of OpenPose + RNN. For simplicity, we refer to this model
as OpenPoseRNN throughout the rest of this readme. And we also thank to Dr.Minh Chuan-Pham and Dr.Quoc Viet-Hoang supported for this project
# Introduction
### Tracking People
Another field that is rapidly developing and showing even greater potential in the future is the detection of cheating among candidates using AI. When operational, this system marks and captures images if it detects any anomalies or violations, sending them to Telegram for verification of the misconduct. The reliability of this system is currently at a credible level and is undergoing further development and testing.

This deep learning-based system is being applied in developed countries worldwide such as the UK, France, the USA, and various Asian countries like Japan, South Korea, among others. Some universities such as Tsinghua University, Peking University, Standford University,... used technology to anti-cheating at examination. It is being implemented in collaboration with examination invigilators to achieve the highest effectiveness and ensure the utmost fairness in examinations.

# Results
### Summary Cheating Recognition ( using OpenPose + Yolov3+ Recurrent Neural Netword)
<p align="center">
    <img src="github/video/video_demo.gif" width="1000">
    <br>
    <sup>Testing with 12422TN class <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose" target="_blank"><i> on OpenPose </i></a>
</p>

### For Human Detection
For this part, we use [YOLOv3](https://github.com/ultralytics/yolov3) to detection human in rooms. To evaluation this model we use trainval35k set, which is slpit from the original [MS-COCO 2017](https://cocodataset.org/#home) dataset. Result shown in Table 1. 

**Table 1. Comparison resukt of human detection in images with 3 other models**
|    Models   | Avg. | Precision |  IoU |
|:-----------:|:----:|:---------:|:----:|
| [Faster-RCNN](https://arxiv.org/abs/1506.01497) | 21.9 |    42.7   |   -  |
|    SSD300   | 25.2 |    43.1   | 26.1 |
|    YOLOv2   | 21.6 |     44    | 19.2 |
|   **Ours**    |**25.3**| **44.5**   | **25.9** |

### For skeleton position localization

For skeleton position localization, we use [OpenPose]("https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detection skeletion human. To evaluate this model we used [MS-COCO2015](https://cocodataset.org/#home) datasets. Result shown in Table 2.

**Table 2. Evaluation results of skeleton position localization compared with 2 other models.**
|     Models    |  AP@0.5  |  AP@0.75 | AP medium | Ap large |
|:-------------:|:--------:|:--------:|:---------:|----------|
| [AlphaPose]() |   89.2   |   79.1   |     69    | 78.6     |
|     Detectron Mask-RCNN    |   25.2   |   43.1   |    26.1   | 68.2     |
|    **Ours**   | **88.0** | **73.1** |  **62.2** | **78.6** |

Beside, we also use FPS and GPU Memory to evluate this. Result shown in Table 3 for Multi people and table 4 for single people.

**Table 3. Results in Multi-people**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   73.4%   |   1.15   |
|    **Ours**   | **21.3%** | **18.39** |

**Table 4. Results in Single-people**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   60.3%   |   23.71   |
|    **Ours**   | **21.3%** | **18.77** |

### For recognition 
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
- You can run this file ```app.py``` to start this project. 
- Input if login  ```username:abc``` and ```password:abc``` to login
- The IP and Port you can access is ```http://localhost:8080/``` or with other laptop or smartphone in same network is ```http://192.168.1.44:8080```
2. To trainning model you read ```Hướng dẫn sử dụng``` in tab ```Tổng quan```

### With Docker
1. Quick Run
You access this project with this command
```bash
docker run -p 8080:8080 [name_you_choose in Installation]
Example: docker run -p 8080:8080 nguyendat135/trackingstudents
```
2. After you can access it with ```http://localhost:8080/``` or with other laptop or smartphone in same network is ```http://192.168.1.44:8080```
3. To trainning model you read ```Hướng dẫn sử dụng``` in tab ```Tổng quan```
# Structures
**Structures for all models**
<p align="center">
    <img src="github/image/structure.png" width="500">
    <br>
</p>

# Send Us FeedBack
Our project is open source for research purposes, and we want to improve it! So let us know (create a new GitHub issue or pull request, email us, etc.) if you...
1. Find/fix any bug (in functionality or speed) or know how to speed up or improve any part of Students Tracking.
2. Want to add/show some cool functionality/demo/project made on top of Students Tracking. We can add your project link to your [Issue](https://github.com/datnguyen-tien204/Tracking_Students/issues)

# Thanks
Thank you for the guidance of PhD. Minh Chuan Pham in the process of creating this project, as well as the evaluation board consisting of PhD. Quoc Viet Hoang and PhD. Dinh Chien Nguyen, who helped us improve the results and provided feedback for this project.

# License
This project is freely available for free non-commercial use. If it useful you can give 1 star. Thanks for using.
