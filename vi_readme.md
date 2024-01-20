![header](github/image/daaf2d95fb02505c0913.jpg)
[English](README.md) | [简体中文](README.zh-CN.md) | [Tiếng việt](README.vietnam-vn.md)

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

Bài báo này cung cấp một cách thực hiện phát hiện bất thường với sự kết hợp giữa OpenPose và RNN. Để đơn giản, chúng tôi gọi mô hình này là OpenPoseRNN trong phần còn lại của readme này. Và chúng tôi cũng cảm ơn Tiến sĩ Phạm Minh Chuẩn và Tiến sĩ Hoàng Quốc Việt đã hỗ trợ cho dự án này. Hệ thống dựa trên học sâu này đang được áp dụng ở các quốc gia phát triển trên toàn thế giới như Anh, Pháp, Hoa Kỳ và các quốc gia châu Á khác như Nhật Bản, Hàn Quốc, Trung Quốc, v.v. Một số trường đại học như Đại học Thanh Hoa, Đại học Bắc Kinh, Đại học Stanford,… đã sử dụng công nghệ để chống gian lận trong kỳ thi. Nó đang được triển khai phối hợp với các giám thị kỳ thi để đạt hiệu quả cao nhất và đảm bảo công bằng tối đa trong các kỳ thi.

# Results
### Tổng quan phát hiện gian lận ( sử dụng OpenPose và Yolov3+ mạng thần kinh hồi quy)
<p align="center">
    <img src="github/video/video_demo.gif" width="1000">
    <br>
    <sup>Kiểm tra với lớp 12422TN class <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose" target="_blank"><i> dùng OpenPose </i></a>
</p>

### Phát hiện người
Trong phần này, nhóm chúng tôi sử dụng [YOLOv3](https://github.com/ultralytics/yolov3) để phát hiện người trong phòng thi. Để đánh giá mô hình này, chúng tôi sử dụng một phần dữ liệu được tách ra từ bộ dữ liệu [MS-COCO 2017](https://cocodataset.org/#home) . Kết quả thể hiện trong bảng 1. 

**Table 1. So sánh mô hình với 2 mô hình khác**
|    Models   | Avg. | Precision |  IoU |
|:-----------:|:----:|:---------:|:----:|
| [Faster-RCNN](https://arxiv.org/abs/1506.01497) | 21.9 |    42.7   |   -  |
|    SSD300   | 25.2 |    43.1   | 26.1 |
|    YOLOv2   | 21.6 |     44    | 19.2 |
|   **Ours**    |**25.3**| **44.5**   | **25.9** |

### Phát hiện vị trí khung xương

Để phát hiện vị trí của các khung xương, nhóm chúng tôi sử dụng [OpenPose]("https://github.com/CMU-Perceptual-Computing-Lab/openpose) để phát hiện khung xương người và trích xuất chúng. Để đánh giá chi phần phát hiện khung xương, nhóm chúng tôi sử dụng bộ dữ liệu [MS-COCO2015](https://cocodataset.org/#home). Kết quả thể hiện trong bảng 1.

**Bảng 2. Đánh giá mô hình phát hiện khung xương với 2 mô hình khác**
|     Models    |  AP@0.5  |  AP@0.75 | AP medium | Ap large |
|:-------------:|:--------:|:--------:|:---------:|----------|
| [AlphaPose]() |   89.2   |   79.1   |     69    | 78.6     |
|     Detectron Mask-RCNN    |   25.2   |   43.1   |    26.1   | 68.2     |
|    **Ours**   | **88.0** | **73.1** |  **62.2** | **78.6** |

Ngoài ra, nhóm chúng tôi cũng sử dụng . Kết quả thể hiện tại bảng 3 cho nhiều người và bảng 4 cho 1 người.

**Bảng 3. Results in Multi-people**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   73.4%   |   1.15   |
|    **Ours**   | **21.3%** | **18.39** |

**Bảng 4. Results in Single-people**
|     Models    |  GPU Memory Usage  |  FPS(Frame Per second  |
|:-------------:|:--------:|:--------:|
| AlphaPose |   60.3%   |   23.71   |
|    **Ours**   | **21.3%** | **18.77** |

### For recognition 
We used [Recurrrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) to classify action of attendance in room. To evaluate this part we use 2 metrics are Confusion Matrix and Receiver operating characteristic (ROC). Result shown in [Figure 1](github/image/confMatrix.jpg) and [Figure 2](github/image/RocCurves)

<p align="center">
    <img src="github/image/confMatrix.jpg" width="1000">
    <br>
    <sup>Fig 1. Result of all label with<a href="https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank"><i> Confusion Matrix  </i></a>
</p>

<p align="center">
    <img src="github/image/RocCurves.png" width="1000">
    <br>
    <sup>Fig 2. Result of all label with<a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc" target="_blank"><i> ROC  </i></a>
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
