import os
import sys
import glob
import cv2
import numpy as np
import torch
from typing import Callable, List

import cvml.detection.dataset.detection_dataset
from cvml.detection.dataset.detection_dataset import DetectionDataset
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.detection.dataset.image_transforming import expo
from cvml.detection.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.detection.dataset.image_transforming import convert_to_mixed


def wrap_expo(img: np.ndarray):
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = expo(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



def create_detection_dataset(
    source_dirs: List[str],
    save_dir: str,
    changes: List[str] = None,
    split_proportions: dict = None,
    install_images: bool = True,
    install_labels: bool = True,
):
    
    final_dataset = DetectionDataset()
    
    for dataset_dir in source_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        image_sources = convert_paths_to_single_sources(paths=image_files,
                                                        preprocess_fn=convert_to_mixed)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = lambda x: os.path.split(dataset_dir)[-1] + '_' + x

        annotation_data = AnnotationConverter.read_coco(annotation_path)
        annotation_data = AnnotationEditor.change_classes_by_new_classes(annotation_data, changes)

        dataset = DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset


    final_dataset.split_by_proportions(split_proportions)
    final_dataset.install(save_dir, install_images, install_labels)




if __name__ == '__main__':

    raw_datasets_dir = '/home/student2/datasets/raw/TMK_3010'
    raw_dirs = glob.glob(os.path.join(raw_datasets_dir, '*SCV3*'))
    raw_dirs.sort()
    
    result_dir = '/home/student2/datasets/prepared/tmk_cvs3_yolov5_03012023'
    
    cls_names = ['other', 'tube', 'sink', 'riska']
    split_proportions = {'train': 0.8, 'valid': 0.2, 'test': 0.0}
    
    create_detection_dataset(raw_dirs, result_dir, cls_names, split_proportions)

