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

from cvml.core.bounding_box import BoundingBox, CoordinatesType, BBType, BBFormat
from cvml.detection.dataset.detection_dataset import DetectionDataset
from cvml.detection.dataset.image_source import ImageSource
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.detection.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.detection.dataset.image_transforming import convert_to_mixed, expo


def main():
    dataset_paths = [
        '',
    ]
    
    for dataset_dir in dataset_paths:
        annot_part1_path = os.path.join(dataset_dir, 'annotations', 'part1.json')
        annot_part2_path = os.path.join(dataset_dir, 'annotations', 'part2.json')
        
        annot_part1 = AnnotationConverter.read_coco(annot_part1_path)
        annot_part2 = AnnotationConverter.read_coco(annot_part2_path)
        
        annot = annot_part1 + annot_part2
        
        AnnotationConverter.write_coco(annot, os.path.join(dataset_dir, 'annotations', 'united.json'))
        

if __name__ == '__main__':
    main()