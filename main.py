# -*- coding: UTF-8 -*-
import cv2 as cv
import time
from utils import choose_run_mode, load_pretrain_model
from OpenPose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
current_struct_time = time.localtime()
formatted_time = time.strftime("%d/%m/%Y %H:%M:%S", current_struct_time)


# Load models
estimator = load_pretrain_model('mobilenet_thin')
action_classifier = load_action_premodel('open_pose2\Action\framewise_recognition_under_scene.h5')

# Initialize parameters
fps_interval = 1
start_time = time.time()
fps_count = 0
frame_count = 0
frame_count2=0
capture_images = False
output_folder = 'ViPham'

#rtsp://admin:AGBSPI@192.168.1.120:554/H.264"
# Choose video source
#cap = choose_run_mode(r"rtsp://admin:AGBSPI@192.168.1.120:554/H.264")
cap = choose_run_mode(0)


while cv.waitKey(1) < 0:
    has_frame, show = cap.read()
    if has_frame:
        fps_count += 1
        frame_count += 1

        humans = estimator.inference(show)
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)
        show,init_label = framewise_recognize(pose, action_classifier)

        height, width = show.shape[:2]
        if (time.time() - start_time) > fps_interval:
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0
            start_time = time.time()

        # Show number of detected humans
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        #Show Frame
        cv.imshow('Phat hien gian lan', show)


# video_writer.release()
cap.release()

