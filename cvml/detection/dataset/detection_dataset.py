import os
import sys
import cv2
import random
import math
import numpy as np
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum
import logging

import time
from cvml.core.bounding_box import BoundingBox
from cvml.detection.dataset.annotation_converter import AnnotationConverter, write_yolo_labels, read_yolo_labels
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.image_source import ImageSource


class LabeledImage:
    def __init__(self,
                 image_source: ImageSource = None,
                 bboxes: List[BoundingBox] = None,
                 name: str = None):

        self.image_source = image_source
        self.bboxes = bboxes or []
        self.name = name or image_source.get_name()

    def save(self, images_dir: str = None, labels_dir: str = None):
        if images_dir is not None and self.image_source is not None:
            self.image_source.save(os.path.join(images_dir, self.name + '.jpg'))

        if labels_dir is not None:
            write_yolo_labels(os.path.join(labels_dir, self.name + '.txt'),
                              self.bboxes)


class DetectionDataset:
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in detection tasks.
    """

    def __init__(self, 
                 labeled_images: List[LabeledImage] = None, 
                 splits: Dict[str, List[int]] = None,
                 classes: List[str] = None):
        """
        :labeled_images: list of labeled images
        :splits: dict of lists of labeled images' indexes, which related to specific split
                 (for example, {'train': [1, 2, 3, 4, 5], 'valid': [6, 7], 'test': [8]})
        """
        self.labeled_images = labeled_images or []
        self.splits = splits or {}
        self.classes = classes or []
        
        self.logger = self._get_logger()

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, item):
        return self.labeled_images[item]

    def __add__(self, other):
        sum_labeled_images = self.labeled_images + other.labeled_images
        
        self_split_names = set(self.splits.keys())
        other_split_names = set(other.splits.keys())
        sum_split_names = self_split_names or other_split_names
        sum_splits = {}
        
        for name in sum_split_names:
            sum_splits[name] = []
            if name in self_split_names:
                sum_splits[name] += self.splits[name]
            if name in other_split_names:
                sum_splits[name] += list(map(lambda x: x + len(self), other.splits[name]))
        
        return DetectionDataset(sum_labeled_images, sum_splits, self.classes)

    def update(self, image_sources: List[ImageSource] = None,
               annotation: Annotation = None):
        
        image_sources = image_sources or []
        annotation = annotation or Annotation()
        self.classes = annotation.classes

        for image_source in image_sources:
            source_name = image_source.get_name()
            if source_name in annotation.bbox_map.keys():
                labels = annotation.bbox_map[source_name]
            else:
                labels = []

            labeled_image = LabeledImage(image_source, labels, source_name)
            self.labeled_images.append(labeled_image)
    
    def rename(self, rename_callback: Callable):
        for i in range(len(self.labeled_images)):
            self.labeled_images[i].name = rename_callback(self.labeled_images[i].name)
            for bb in self.labeled_images[i].bboxes:
                bb._image_name = rename_callback(bb._image_name)

    def split_by_proportions(self, proportions: dict):
        all_idx = [i for i in range(len(self.labeled_images))]
        random.shuffle(all_idx)

        length = len(self.labeled_images)
        split_start_idx = 0
        split_end_idx = 0

        # Reset current split indexes
        self.splits = {}

        num_of_names = len(proportions.keys())

        for i, split_name in enumerate(proportions.keys()):
            split_end_idx += math.ceil(proportions[split_name] * length)
            self.splits[split_name] = all_idx[split_start_idx: split_end_idx]
            split_start_idx = split_end_idx

            if i + 1 == num_of_names and split_end_idx < len(all_idx):
                self.splits[split_name] += all_idx[split_end_idx: len(all_idx)]
        
        # logging
        message = "In dataset the following splits was created: "
        for i, split_name in enumerate(self.splits.keys()):
            message += f"{split_name}({len(self.splits[split_name])})"
            if i != len(self.splits.keys()) - 1:
                message += ", "
        self.logger.info(message)

    def split_by_dataset(self, yolo_dataset_path: str):

        # Define names of splits as dirnames in dataset directory
        split_names = [name for name in os.listdir(yolo_dataset_path)
                       if os.path.isdir(os.path.join(yolo_dataset_path, name))]

        # Reset current split indexes
        self.splits = {}

        for split_name in split_names:

            # Place names of orig dataset split in set structure
            orig_dataset_files = os.listdir(os.path.join(yolo_dataset_path, split_name, 'labels'))
            orig_names_set = set()

            for file in orig_dataset_files:
                name, ext = os.path.splitext(file)
                orig_names_set.add(name)

            # If new_name in orig dataset split then update split indexes of current dataset
            self.splits[split_name] = []
            for i, labeled_image in enumerate(self.labeled_images):
                new_name = labeled_image.name
                if new_name in orig_names_set:
                    self.splits[split_name].append(i)

    def install(self, 
                dataset_path: str, 
                install_images: bool = True, 
                install_labels: bool = True, 
                install_annotations: bool = True, 
                install_description: bool = True):
        
        for split_name in self.splits.keys():
            split_idx = self.splits[split_name]

            os.makedirs(os.path.join(dataset_path, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split_name, 'labels'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split_name, 'annotations'), exist_ok=True)

            images_dir = os.path.join(dataset_path, split_name, 'images') if install_images else None
            labels_dir = os.path.join(dataset_path, split_name, 'labels') if install_labels else None
            
            split_bboxes_map = {}
            for i in split_idx:
                self.labeled_images[i].save(images_dir, labels_dir)
                
                if install_annotations:
                    name = self.labeled_images[i].name
                    bboxes = self.labeled_images[i].bboxes
                    split_bboxes_map[name] = bboxes
                
                self.logger.info(f"In split \"{split_name}\" image \"{self.labeled_images[i].name}\" is processed")
            
            if install_annotations:
                split_annotation = Annotation(self.classes, split_bboxes_map)
                coco_path = os.path.join(dataset_path, split_name, 'annotations', f'{split_name}.json')
                AnnotationConverter.write_coco(split_annotation, coco_path)
            
            if install_description:
                self._write_description(os.path.join(dataset_path, 'data.yaml'))
    
    def exclude_by_names(self, excluding_names: Set[str], splits: List[str]):
        
        for split in splits:
            for i in range(len(self.splits[split]) - 1, -1, -1):
                idx = self.splits[split][i]
                labeled_image = self.labeled_images[idx]
                new_name = labeled_image.name

                if new_name in excluding_names:
                    self.splits[split].pop(i)
    
    def _get_logger(self) -> logging.Logger:
        
        logger = logging.getLogger(__name__)
        
        # logger.handlers = []
        logger.setLevel(logging.DEBUG)
        
        # # Create handlers
        # s_handler = logging.StreamHandler(sys.stdout)
        # s_handler.setLevel(logging.INFO)
        
        # # Create formatters and add it to handlers
        # s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # s_handler.setFormatter(s_format)
        
        # # Add handlers to the logger
        # logger.addHandler(s_handler)
        
        return logger
    
    def _write_description(self, path: str):
        text = f"train: ../train/images\n" \
               f"val: ../valid/images\n\n" \
               f"nc: {len(self.classes)}\n" \
               f"names: {self.classes}"
        with open(path, 'w') as f:
            f.write(text)
        





