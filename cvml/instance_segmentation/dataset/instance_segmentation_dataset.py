import os
import cv2
import random
import math
import numpy as np
from itertools import groupby
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum

from cvml.core.bounding_box import BoundingBox
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.detection_dataset import DetectionDataset, LabeledImage
from cvml.detection.dataset.image_source import ImageSource, MultipleImageSource


def convert_mask_to_coco_rle(color_mask: np.ndarray, bbox: BoundingBox) -> dict:
    x, y, w, h = map(int, bbox.get_absolute_bounding_box())
    width, height = color_mask.shape[:2]

    rle = {
        'size': [width, height],
        'counts': [],
    }

    x = min(x, width)
    y = min(y, height)
    w = min(w, width - x)
    h = min(h, height - y)

    if w == 0 or h == 0:
        rle['counts'] = [0]
        return rle

    obj_crop = color_mask[y: y + h, x: x + w]
    obj_crop = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2GRAY)
    ret, binary_obj_crop = cv2.threshold(obj_crop, 1, 1, cv2.THRESH_BINARY)

    binary_mask = np.zeros((height, width), dtype='uint8')
    binary_mask[y: y + h, x: x + w] = binary_obj_crop

    counts = []

    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    
    rle['counts'] = counts

    return rle


class ISImageSource(MultipleImageSource):
    def __init__(self, paths: List[str], color_mask_path: str, main_channel: int = 0, preprocess_fns: List[Callable] = None):
        super(ISImageSource, self).__init__(paths, main_channel, preprocess_fns)
        self.color_mask_path = color_mask_path
    
    def get_color_mask(self):
        img = cv2.imread(self.color_mask_path)
        return img


class ISLabeledImage(LabeledImage):
    def __init__(self,
                 image_source: ImageSource = None,
                 bboxes: List[BoundingBox] = None,
                 name: str = None):
        
        super(ISLabeledImage, self).__init__(image_source, bboxes, name)

    def save(self, images_dir: str = None):
        if images_dir is not None and self.image_source is not None:
            self.image_source.save(os.path.join(images_dir, self.name + '.jpg'))



class ISDataset(DetectionDataset):
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in instance segmentation tasks.
    """

    def __init__(self,
                 labeled_images: List[LabeledImage] = None, 
                 splits: Dict[str, List[int]] = None):
        
        super(ISDataset, self).__init__(labeled_images, splits)
        self.classes = []
    
    def __add__(self, other):
        sum_labeled_images = self.labeled_images + other.labeled_images
        sum_classes = self.classes
        
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
        
        dataset = ISDataset(sum_labeled_images, sum_splits)
        dataset.classes = sum_classes
        return dataset

    def update(self, image_sources: List[ISImageSource] = None,
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
            
            # add segmentation
            color_mask = image_source.get_color_mask()
            if color_mask is not None:
                for bbox in labels:
                    rle = convert_mask_to_coco_rle(color_mask, bbox)
                    bbox.set_segmentation(rle)
            else:
                labels = []
                # for bbox in labels:
                #     width, height = bbox.get_image_size()
                #     rle = {
                #         'size': [width, height],
                #         'counts': [width * height],
                #     }
                #     print(bbox.get_image_name(), rle)
                #     bbox.set_segmentation(rle)
                    
            labeled_image = ISLabeledImage(image_source, labels, source_name)
            self.labeled_images.append(labeled_image)
            print(source_name)

    def install(self, dataset_path: str, install_images: bool = True, install_annotations: bool = True):
        
        converter = AnnotationConverter()
        for split_name in self.splits.keys():
            split_idx = self.splits[split_name]

            images_dir = os.path.join(dataset_path, split_name, 'images')
            annotations_dir = os.path.join(dataset_path, split_name, 'annotations')

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)

            split_bboxes = []

            for i in split_idx:
                print(self.labeled_images[i].name)
                if install_images:
                    self.labeled_images[i].save(images_dir)

                split_bboxes += self.labeled_images[i].bboxes

            annotation = converter.read_bboxes(split_bboxes, self.classes)
            if install_annotations:
                converter.write_coco(annotation, os.path.join(annotations_dir, 'data.json'))





