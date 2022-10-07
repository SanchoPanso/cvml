import os
import cv2
import random
import math
import numpy as np
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum

from data_science_tools.core.bounding_box import BoundingBox
from data_science_tools.core.annotation import Annotation
from data_science_tools.detection.dataset.detection_dataset import DetectionDataset, LabeledImage
from data_science_tools.detection.dataset.image_sources import ImageSource


class ISLabeledImage(LabeledImage):
    def __init__(self,
                 image_source: ImageSource = None,
                 bboxes: List[BoundingBox] = None,
                 segmentation: List[List[List[float]]] = None,
                 name: str = None):
        
        super(ISLabeledImage, self).__init__(image_source, bboxes, name)
        self.segmentation = segmentation or []

    def save(self, images_dir: str = None):
        if images_dir is not None and self.image_source is not None:
            self.image_source.save_to(os.path.join(images_dir, self.name + '.jpg'))



class ISDataset(DetectionDataset):
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in instance segmentation tasks.
    """

    def __init__(self,
                 labeled_images: List[LabeledImage] = None, 
                 splits: Dict[str, List[int]] = None):
        
        super(ISDataset, self).__init__(labeled_images, splits)

    def update(self, image_sources: List[ImageSource] = None,
               annotation: Annotation = None):
        
        image_sources = image_sources or []
        annotation = annotation or Annotation()

        for image_source in image_sources:
            source_name = image_source.get_name()
            if source_name in annotation.bounding_boxes.keys():
                labels = annotation.bounding_boxes[source_name]
            else:
                labels = []

            labeled_image = LabeledImage(image_source, labels, source_name)
            self.labeled_images.append(labeled_image)

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

    def install(self, dataset_path: str, install_images: bool = True, install_labels: bool = True):
        for split_name in self.splits.keys():
            split_idx = self.splits[split_name]

            os.makedirs(os.path.join(dataset_path, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split_name, 'labels'), exist_ok=True)

            for i in split_idx:
                images_dir = os.path.join(dataset_path, split_name, 'images')
                labels_dir = os.path.join(dataset_path, split_name, 'labels')
                print(self.labeled_images[i].name)

                images_dir = images_dir if install_images else None
                labels_dir = labels_dir if install_labels else None
                self.labeled_images[i].save(images_dir, labels_dir)




