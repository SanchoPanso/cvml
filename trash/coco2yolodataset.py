import os
import cv2
import random
import math
import numpy as np
from abc import ABC
from typing import List, Set, Callable
from enum import Enum

from ..cvml.detection.dataset.io_handling import write_yolo_labels, read_yolo_labels
from ..cvml.detection.dataset.extractor import Extractor
from ..cvml.detection.dataset.image_sources import ImageSource


class LabeledImage:
    def __init__(self,
                 image_source: ImageSource,
                 labels: List[List[float]] = None,
                 new_name: str = None):

        self.image_source = image_source
        self.labels = labels
        self.new_name = new_name

    def save(self, images_dir: str = None, labels_dir: str = None):
        if images_dir is not None:
            img = self.image_source.read()

            # cv2.imwrite(os.path.join(images_dir, self.new_name + '.jpg'), img)
            is_success, im_buf_arr = cv2.imencode(".jpg", img)
            im_buf_arr.tofile(os.path.join(images_dir, self.new_name + '.jpg'))

        if labels_dir is not None:
            write_yolo_labels(os.path.join(labels_dir, self.new_name + '.txt'),
                              self.labels)


class Coco2YoloDataset:
    def __init__(self):
        self.labeled_images = []
        self.splits = {}

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, item):
        return self.labeled_images[item]

    def add(self,
            image_sources: List[ImageSource] = None,
            annotation_data: dict = None,
            rename_callback: Callable = None):

        # Necessary data is absent - leave attribute empty and return
        if annotation_data is None or image_sources is None:
            return

        annotation_names = annotation_data['annotations'].keys()

        for image_source in image_sources:
            # if *** is not correspond any image file, then skip it
            name = image_source.get_name()
            if name not in annotation_names:
                continue

            new_name = rename_callback(name) if rename_callback is not None else None
            labels = annotation_data['annotations'][name]

            labeled_image = LabeledImage(image_source, labels, new_name)
            self.labeled_images.append(labeled_image)

    def split_by_proportions(self, proportions: dict):
        all_idx = [i for i in range(len(self.labeled_images))]
        random.shuffle(all_idx)     # check

        length = len(self.labeled_images)
        split_start_idx = 0
        split_end_idx = 0

        self.splits = {}
        num_of_names = len(proportions.keys())
        for i, split_name in enumerate(proportions.keys()):
            split_end_idx += math.ceil(proportions[split_name] * length)
            self.splits[split_name] = all_idx[split_start_idx: split_end_idx]
            split_start_idx = split_end_idx

            if i + 1 == num_of_names and split_end_idx < len(all_idx):
                self.splits[split_name] += all_idx[split_end_idx: len(all_idx)]

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
                new_name = labeled_image.new_name
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
                print(self.labeled_images[i].new_name)

                images_dir = images_dir if install_images else None
                labels_dir = labels_dir if install_labels else None
                self.labeled_images[i].save(images_dir, labels_dir)
    
    def exclude_by_new_names(self, excluding_new_names: Set[str], splits: List[str]):
        
        for split in splits:
            for i in range(len(self.splits[split]) - 1, -1, -1):
                idx = self.splits[split][i]
                labeled_image = self.labeled_images[idx]
                new_name = labeled_image.new_name

                if new_name in excluding_new_names:
                    self.splits[split].pop(i)
        





