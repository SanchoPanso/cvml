import os
import sys
import glob
import cv2
import numpy as np
import torch
import logging
from typing import Callable, List
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cvml.annotation.bounding_box import BoundingBox, CoordinatesType, BBType, BBFormat
from cvml.dataset.detection_dataset import DetectionDataset
from cvml.dataset.image_source import ImageSource
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.dataset.image_transforming import convert_to_mixed, expo

from cvml.detection.augmentation.golf_augmentation import MaskMixup, MaskMixupAugmentation



def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number2')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number3')))

    # raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'number_december1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'number_december2')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_october1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_january_Marina1')))

    
    save_dir = '/home/student2/datasets/prepared/tmk_number_resnet_07022023' #'/home/student2/datasets/prepared/TMK_CVS3_0701'
    for i in range(9):
        os.makedirs(os.path.join(save_dir, str(i)), exist_ok=True)

    for raw_dir in raw_dirs:
        images_dir = os.path.join(raw_dir, 'images')
        annotations_path = os.path.join(raw_dir, 'annotations', 'digits.json')
        annotation = AnnotationConverter.read_coco(annotations_path)

        for name in annotation.bbox_map:
            bboxes = annotation.bbox_map[name]
            img_path = os.path.join(images_dir, f"{name}.png")

            if not os.path.exists(img_path):
                continue
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            for i, bbox in enumerate(bboxes):
                cls_id = bbox.get_class_id()
                x, y, w, h = map(int, bbox.get_coordinates())
                crop = img[y: y + h, x: x + w]
                save_name = f"{os.path.split(raw_dir)[-1]}_{name}_{i + 1}.png"

                print(save_name)
                ext = os.path.splitext(os.path.split(save_name)[-1])[1]
                is_success, im_buf_arr = cv2.imencode(ext, img)
                im_buf_arr.tofile(save_name)

        

if __name__ == '__main__':
    main()