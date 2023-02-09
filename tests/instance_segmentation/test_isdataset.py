import pytest
import os
import sys
import json
import numpy as np
import cv2

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)

from cvml.annotation.bounding_box import BBType, BoundingBox, CoordinatesType, BBFormat
from cvml.annotation.annotation import Annotation
from cvml.annotation.annotation_converting import write_coco, write_yolo
from cvml.annotation.annotation_converting import read_coco, read_yolo
from cvml.dataset.instance_segmentation_dataset import convert_mask_to_coco_rle


def test_convert_mask_to_coco_rle():
    img_shape_small = (10, 10, 3)
    img_shape_big = (20, 20, 3)
    color = (0, 0, 255)

    img1 = np.zeros(img_shape_small, dtype='uint8')

    img2 = np.zeros(img_shape_small, dtype='uint8')
    img2 = cv2.rectangle(img2, (2, 0), (8, 10), color, -1)

    img3 = np.zeros(img_shape_small, dtype='uint8')
    img3 = cv2.rectangle(img3, (0, 0), (8, 10), color, -1)

    img4 = np.zeros(img_shape_big, dtype='uint8')
    img4 = cv2.rectangle(img4, (0, 1), (20, 20), color, -1) 

    img5 = np.zeros((0, 0, 3), dtype='uint8')

    imgs = [img1, img2, img3, img4, img5]
    counts = [
        [100],
        [20, 70, 10],
        [0, 90, 10],
        [1] + [9, 11] * 9 + [9, 210],
        [0],
    ]

    bbox = BoundingBox(0, 0, 0, 10, 10)

    for i in range(len(imgs)):
        got_counts = convert_mask_to_coco_rle(imgs[i], bbox)['counts']
        expected_counts = counts[i]
        assert got_counts == expected_counts




