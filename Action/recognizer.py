# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from pathlib import Path
from Detection.deep_sort import preprocessing
from Detection.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Detection.deep_sort.detection import Detection
from Detection import generate_dets as gdet
from Detection.deep_sort.tracker import Tracker
from keras.models import load_model
from .action_enum import Actions
import os


# Use Deep-sort(Simple Online and Realtime Detection)
# To track multi-person for multi-person actions recognition

file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort
model_filename = str(file_path/'Detection/graph_model/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box
trk_clr = (0, 255, 0)
trk_clr_operate = (0, 0, 255)


def load_action_premodel(model):
    return load_model(model)


def framewise_recognize(pose, pretrained_model):
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])
    global init_label
    init_label=None
    frame_count=0

    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        # 
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        #tracker
        tracker.predict()
        tracker.update(detections)

        # track，bounding boxes_ID
        trk_result = []
        for trk in tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
            # 标注track_ID
            trk_id = 'ID-' + str(trk.track_id)
            cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 2)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                j = 0
            if joints_norm_per_frame.size > 0:
                joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
                init_label = Actions(pred).name
                cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 2)
                # Action
                if init_label == 'NemPhao':
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr_operate, 2)
                elif init_label=="QuayTrai":
                    #(255 165 0)
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), (255,165,0), 2)
                elif init_label=="QuayPhai":
                    #(238 18 137)
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), (238, 18, 137), 2)
                elif init_label=="DungDay":
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), (0,255,255), 2)
                elif init_label=="QuaySau":
                    #(238 232 170)
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), (238,232,170), 2)
                else:
                    cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
                cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 2)

    return frame,init_label

