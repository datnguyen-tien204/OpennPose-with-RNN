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
<p align="center">
    <img src="github/images_introduction/cheating.jpg" width="360">
</p>
Another field that is rapidly developing and showing even greater potential in the future is the detection of cheating among candidates using AI. When operational, this system marks and captures images if it detects any anomalies or violations, sending them to Telegram for verification of the misconduct. The reliability of this system is currently at a credible level and is undergoing further development and testing.

This deep learning-based system is being applied in developed countries worldwide such as the UK, France, the USA, and various Asian countries like Japan, South Korea, among others. It is being implemented in collaboration with examination invigilators to achieve the highest effectiveness and ensure the utmost fairness in examinations.

# Results

### Cheating Recognition ( using OpenPose)
<p align="center">
    <img src="github/video/5533218175348215452 (1)" width="1000">
    <br>
    <sup>Testing with 12422TN class <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose" target="_blank"><i> on OpenPose </i></a>
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
``` bash
Tracking_Students
+---.idea
¦   +---inspectionProfiles
+---Action
¦   +---training
¦   +---__pycache__
+---Auth
¦   +---__pycache__
+---Dataset
¦   +---FaceData
+---graph_models
¦   +---mobilenet_thin
¦   +---VGG_origin
+---Models
+---Pose
¦   +---graph_models
¦   ¦   +---mobilenet_thin
¦   ¦   +---VGG_origin
¦   +---__pycache__
+---profile_detection
¦   +---haarcascades
¦   +---__pycache__
+---src
¦   +---Action
¦   ¦   +---training
¦   ¦   +---__pycache__
¦   +---align
¦   ¦   +---__pycache__
¦   +---Auth
¦   ¦   +---__pycache__
¦   +---FCRN
¦   +---generative
¦   ¦   +---models
¦   +---models
¦   +---Pose
¦   ¦   +---graph_models
¦   ¦   ¦   +---mobilenet_thin
¦   ¦   ¦   +---VGG_origin
¦   ¦   +---__pycache__
¦   +---QSTP
¦   +---SoNguoi
¦   +---Tracking
¦   ¦   +---deep_sort
¦   ¦   ¦   +---__pycache__
¦   ¦   +---graph_model
¦   ¦   +---__pycache__
¦   +---ViPham
¦   +---__pycache__
+---test
+---test_out
+---Tracking
¦   +---deep_sort
¦   ¦   +---__pycache__
¦   +---graph_model
¦   +---__pycache__
+---trained
+---ViPham
+---__pycache__
```

# Send Us FeedBack
Our project is open source for research purposes, and we want to improve it! So let us know (create a new GitHub issue or pull request, email us, etc.) if you...
1. Find/fix any bug (in functionality or speed) or know how to speed up or improve any part of Students Tracking.
2. Want to add/show some cool functionality/demo/project made on top of Students Tracking. We can add your project link to your [Issue](https://github.com/datnguyen-tien204/Tracking_Students/issues)

# Thanks
Thank you for the guidance of PhD. Minh Chuan Pham in the process of creating this project, as well as the evaluation board consisting of PhD. Quoc Viet Hoang and PhD. Dinh Chien Nguyen, who helped us improve the results and provided feedback for this project.

# License
This project is freely available for free non-commercial use. If it useful you can give 1 star. Thanks for using.
