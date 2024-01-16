# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
import time
import csv
from utils import choose_run_mode, load_pretrain_model#, set_video_writer
from OpenPose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
current_struct_time = time.localtime()
formatted_time = time.strftime("%d/%m/%Y %H:%M:%S", current_struct_time)
baseUrl=''


# Load models
estimator = load_pretrain_model('mobilenet_thin')
action_classifier = load_action_premodel('Action/framewise_recognition.h5')

# Initialize parameters
fps_interval = 1
start_time = time.time()
fps_count = 0
frame_count = 0
frame_count2=0
capture_images = False
output_folder = 'ViPham'

# dongtac = input("Hay nhap dong tac: ")
dongtac='DangTayPhai'

# Choose video source
cap = choose_run_mode(r'D:\Download All\download12\Compressed\open_pose2\video\DangTayPhai.mp4')
# video_writer = set_video_writer(cap, write_fps=int(7.0))

# Initialize joint data file
f = open(f'{dongtac}.txt', 'a+')
# f.write('nose_x,nose_y,neck_x,neck_y,Rshoulder_x,Rshoulder_y,Relbow_x,Relbow_y,Rwrist_x,RWrist_y,LShoulder_x,LShoulder_y,LElbow_x,LElbow_y,LWrist_x,LWrist_y,RHip_x,RHip_y,RKnee_x,RKnee_y,RAnkle_x,RAnkle_y,LHip_x,LHip_y,LKnee_x,LKnee_y,LAnkle_x,LAnkle_y,REye_x,REye_y,LEye_x,LEye_y,REar_x,REar_y,LEar_x,LEar_y,class')
# f.write('\n')

csv_file = open(f'{dongtac}.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)




while cv.waitKey(1) < 0:
    has_frame, show = cap.read()
    if has_frame:
        fps_count += 1
        frame_count += 1

        humans = estimator.inference(show)
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)
        show,init_label = framewise_recognize(pose, action_classifier)

        height, width = show.shape[:2]
        # Calculate and show FPS
        if (time.time() - start_time) > fps_interval:
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0
            start_time = time.time()
        # fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        # cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv.resize(show,(800,600))


        cv.imshow('Phat hien gian lan', show)

        joints_norm_per_frame = np.array(pose[-1]).astype(np.str_)


        strluu = (joints_norm_per_frame)
        writer.writerow(strluu)
    else:
        break



cap.release()
csv_file.close()

