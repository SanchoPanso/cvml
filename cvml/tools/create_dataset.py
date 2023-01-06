import os
import sys
import glob
import cv2
import numpy as np
import torch
from typing import Callable, List
import argparse

from cvml.detection.dataset.detection_dataset import DetectionDataset
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.detection.dataset.image_transforming import expo
from cvml.detection.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.detection.dataset.image_transforming import convert_to_mixed


def create_tubes_detection_dataset(
    source_dirs: List[str],
    save_dir: str,
    classes: List[str] = None,
    sample_proportions: dict = None,
    install_images: bool = True,
    install_labels: bool = True,
    install_annotations: bool = True,
    install_description: bool = True,
    create_compressed_samples: bool = True,
):
    """Create special detection dataset, where images preprocessed by polarization algorythm (convert_to_mixed),
    renamed by scheme f"{source_dataset_dir}_{source_image_name}" and converted in ".jpg" format.

    :param source_dirs: list of source dirs in special format. 
                        source_dir_name
                        |-images
                        |-annotations
                          |-instances_default.json
                        
                        images - subdir with original images
                        annotations/instances_default.json - annotaion in coco-format
    
    :param save_dir: path to dir for saving dataset
    :param classes: classes, chosen from annotation and renumbered, defaults to None (choosing all classes)
    :param sample_proportions: dict with keys - names of samples, values - indexes of images in sample, defaults to None
    :param install_images: _description_, defaults to True
    :param install_labels: _description_, defaults to True
    :param install_annotations: _description_, defaults to True
    :param install_description: _description_, defaults to True
    :param create_compressed_samples: _description_, defaults to True
    """
    
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
        classes = classes or annotation_data.classes
        annotation_data = AnnotationEditor.change_classes_by_new_classes(annotation_data, classes)

        dataset = DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    sample_proportions = sample_proportions or {}
    final_dataset.split_by_proportions(sample_proportions)
    final_dataset.install(save_dir, install_images, install_labels, install_annotations, install_description)
    
    if create_compressed_samples:
        if os.name == 'posix':
            for sample_name in sample_proportions:
                sample_path = os.path.join(save_dir, sample_name)
                os.system(f"zip -r {sample_path}.zip {sample_path}")
                os.system(f"split {sample_path}.zip {sample_path}.zip.part_ -b 999MB")
        elif os.name == 'nt':
            os.system("echo Zip not Implemented")    # TODO
        else:
            pass

