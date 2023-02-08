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



def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*cvs1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*csv1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*number*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))

    raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, '*comet*')))
    raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_comet_december3')))
    raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_january_Marina1')))
    
    save_dir = '/home/student2/datasets/crops/crop_number_07022023'
    os.makedirs(save_dir, exist_ok=True)
    
    crop_cnt = 1
    
    for dataset_dir in raw_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        annotation_data = AnnotationConverter.read_coco(annotation_path)
        
        for img_name in annotation_data.bbox_map:
            img_path = os.path.join(image_dir, img_name + '.png')
            if not os.path.exists(img_path):
                continue
            
            mask_path = os.path.join(image_dir, img_name + '_color_mask.png')
            if not os.path.exists(mask_path):
                continue
            
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = convert_to_mixed(img)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            
            bboxes = annotation_data.bbox_map[img_name]
            
            for bbox in bboxes:
                if annotation_data.classes[bbox.get_class_id()] != 'number':
                    continue
                x, y, w, h = map(int, bbox.get_coordinates())
                img_crop = img[y: y + h, x: x + w]
                mask_crop = mask[y: y + h, x: x + w]
                masked_img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                
                cv2.imwrite(os.path.join(save_dir, str(crop_cnt) + '.png'), masked_img_crop)
                
                print(crop_cnt)
                crop_cnt += 1
            
            
    
    
    
if __name__ == '__main__':
    main()