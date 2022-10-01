import torch
import numpy as np
import os
import cv2
import time
from typing import List

from data_science_tools.core.detector import Detector
from data_science_tools.core.bounding_box import BoundingBox, BBType


class Yolov5Detector(Detector):
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    'custom',
                                    path=weights_path,
                                    # force_reload=True,
                                    map_location=device)

    def __call__(self, img: np.ndarray, size=640, conf=0.25, iou=0.45) -> List[BoundingBox]:

        self.model.conf = conf  # NMS confidence threshold
        self.model.iou = iou    # NMS IoU threshold

        results = self.model(img, size=size)

        df = results.pandas().xyxy[0]
        xmin = np.array(df['xmin'])
        xmax = np.array(df['xmax'])
        ymin = np.array(df['ymin'])
        ymax = np.array(df['ymax'])

        cls_conf = np.array([df['confidence']]).T
        cls_ids = np.array([df['class']]).T

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        bounding_boxes = []
        for i in range(x.shape[0]):
            bb = BoundingBox(cls_ids[i], 
                             x[i], 
                             y[i], 
                             w[i], 
                             h[i], 
                             cls_conf[i], 
                             bb_type=BBType.Detected)
            bounding_boxes.append(bb)

        return bounding_boxes



