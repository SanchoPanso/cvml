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
from data_science_tools.detection.dataset.annotation_converter import AnnotationConverter
from data_science_tools.detection.dataset.detection_dataset import DetectionDataset, LabeledImage
from data_science_tools.detection.dataset.image_sources import ImageSource, MultipleImageSource


def convert_mask_to_coco_rle(color_mask: np.ndarray, bbox: BoundingBox) -> dict:
    x, y, w, h = bbox.get_absolute_bounding_box()
    obj_crop = color_mask[y: y + h, x: x + w]
    obj_crop = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2GRAY)
    binary_obj_crop = cv2.threshold(obj_crop, 1, 255, cv2.THRESH_BINARY)

    width, height = color_mask.shape[:2]
    submask = np.zeros((height, width), dtype='uint8')
    submask[y: y + h, x: x + w] = binary_obj_crop

    counts = []
    count = 0
    state = 0
    for x in range(width):
        for y in range(height):
            if submask[y][x] == state:
                count += 1
            else:
                counts.append(count)
                count = 1
                state = submask[y][x]
    
    rle = {
        'size': [width, height],
        'counts': counts,
    }

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

    def update(self, image_sources: List[ISImageSource] = None,
               annotation: Annotation = None):
        
        image_sources = image_sources or []
        annotation = annotation or Annotation()

        for image_source in image_sources:
            source_name = image_source.get_name()
            if source_name in annotation.bounding_boxes.keys():
                labels = annotation.bounding_boxes[source_name]
            else:
                labels = []
            
            # add segmentation
            color_mask = image_source.get_color_mask()
            for bbox in labels:
                rle = convert_mask_to_coco_rle(color_mask, bbox)
                bbox.set_segmentation(rle)

            labeled_image = ISLabeledImage(image_source, labels, source_name)
            self.labeled_images.append(labeled_image)

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

            annotation = converter.read_bboxes(split_bboxes)
            if install_annotations:
                converter.write_coco(annotation)





