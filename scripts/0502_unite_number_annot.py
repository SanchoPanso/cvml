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


def create_annot():
    dataset_paths = [
        '/home/student2/datasets/raw/23_06_2021_номера_оправок_командир',
        '/home/student2/datasets/raw/cvs1_number1',
        '/home/student2/datasets/raw/cvs1_number2',
        '/home/student2/datasets/raw/cvs1_number3',
    ]
    
    for dataset_dir in dataset_paths:
        print(dataset_dir)

        annot_part1_path = os.path.join(dataset_dir, 'annotations', 'digits_part1.json')
        annot_part2_path = os.path.join(dataset_dir, 'annotations', 'digits_part2.json')
        
        annot_part1 = AnnotationConverter.read_coco(annot_part1_path)
        annot_part2 = AnnotationConverter.read_coco(annot_part2_path)
        
        classes = annot_part1.classes
        bbox_map = {}
        for name in annot_part1.bbox_map:
            bbox_map[name] = annot_part1.bbox_map[name]
        for name in annot_part2.bbox_map:
            bbox_map[name] = annot_part2.bbox_map[name]

        annot = Annotation(classes, bbox_map)

        AnnotationConverter.write_coco(annot, os.path.join(dataset_dir, 'annotations', 'digits.json'))

def check_annot():
    dataset_paths = [
        '/home/student2/datasets/raw/23_06_2021_номера_оправок_командир',
        '/home/student2/datasets/raw/cvs1_number1',
        '/home/student2/datasets/raw/cvs1_number2',
        '/home/student2/datasets/raw/cvs1_number3',
    ]
    
    for dataset_dir in dataset_paths:
        print(dataset_dir)
        cnt = 0

        annot_digits_path = os.path.join(dataset_dir, 'annotations', 'digits.json')
        annot_number_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        
        annot_digits = AnnotationConverter.read_coco(annot_digits_path)
        annot_number = AnnotationConverter.read_coco(annot_number_path)
        
        for name in annot_number.bbox_map:
            if name not in annot_digits.bbox_map:
                cnt += 1
                continue
            num_of_digits = len(annot_digits.bbox_map[name])
            num_of_numbers = len([bb for bb in annot_number.bbox_map[name] if annot_number.classes[bb.get_class_id()] == 'number'])

            if num_of_digits != num_of_numbers:
                cnt += 1
        print(cnt)

if __name__ == '__main__':
    create_annot()
    check_annot()