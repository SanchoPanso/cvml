import torch
import numpy as np
import os
import cv2
import time


class Yolov5Detector:
    def __init__(self, weights_path: str):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    'custom',
                                    path=weights_path,
                                    force_reload=True)
        self.duration = 0

    def __call__(self, img: np.ndarray,
                 count_part_x=1, count_part_y=1,
                 size=640, conf=0.25, iou=0.45):

        height, width = img.shape[0:2]
        height_crop = int(height / count_part_y)
        width_crop = int(width / count_part_x)
        det_grapes = []

        # cropping original image count_part_x*count_part_y parties and detection every parties
        for part_x in range(count_part_x):
            for part_y in range(count_part_y):
                x0_crop = part_x * width_crop
                y0_crop = part_y * height_crop
                img_crop = img[y0_crop: y0_crop + height_crop,
                           x0_crop: x0_crop + width_crop]

                dets = self.inference(img_crop, size, conf, iou)  # xywh
                dets[:, :4] = dets[:, :4].astype(int)

                dets[:, 0] += x0_crop
                dets[:, 1] += y0_crop

                if dets.shape[0] != 0 and len(dets.shape) > 1:
                    det_grapes.append(dets)
        if len(det_grapes) == 0:
            return np.zeros((0, 6))
        return np.concatenate(det_grapes, axis=0)

    def inference(self, img: np.ndarray, size=640, conf=0.25, iou=0.45) -> np.array:

        self.model.conf = conf  # NMS confidence threshold
        self.model.iou = iou  # NMS IoU threshold

        self.duration = time.time()
        results = self.model(img, size=size)
        self.duration = time.time() - self.duration

        df = results.pandas().xyxy[0]

        xmin = np.array(df['xmin'])
        xmax = np.array(df['xmax'])
        ymin = np.array(df['ymin'])
        ymax = np.array(df['ymax'])

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        bbox_xywh = np.stack([x, y, w, h]).T

        cls_conf = np.array([df['confidence']]).T
        cls_ids = np.array([df['class']]).T

        if bbox_xywh.shape[0] == 0:
            return np.zeros((0, 6))

        output = np.concatenate([bbox_xywh, cls_conf, cls_ids], axis=1)

        return output

