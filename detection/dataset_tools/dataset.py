import os
import cv2
import random
import math
import numpy as np
from abc import ABC
from typing import List, Callable

from .io_handling import write_yolo_labels, read_yolo_labels
from .extractor import Extractor


class LabeledImage(ABC):
    def save(self, images_dir: str = None, labels_dir: str = None):
        raise NotImplementedError


class DefaultLabeledImage(LabeledImage):
    def __init__(self,
                 image_path: str,
                 annotation_path: str = None,
                 annotation_data: list = None,
                 new_name: str = None):

        self.image_path = image_path
        self.annotation_path = annotation_path
        self.annotation_data = annotation_data
        self.new_name = new_name

    def save(self, images_dir: str = None, labels_dir: str = None):
        if images_dir is not None:
            img = cv2.imread(self.image_path)
            cv2.imwrite(os.path.join(images_dir, self.new_name + '.jpg'), img)

        if labels_dir is not None:
            write_yolo_labels(os.path.join(images_dir, self.new_name + '.txt'),
                              self.annotation_data)


class SplittedLabeledImage(LabeledImage):
    def __init__(self,
                 image_paths: List[str],
                 preprocessers: List[Callable] = None,
                 main_channel: int = 2,
                 annotation_path: str = None,
                 annotation_data: list = None,
                 new_name: str = None):

        self.image_paths = image_paths
        self.preprocessers = [lambda x: x] * 3 if preprocessers is None else preprocessers
        self.main_channel = main_channel
        self.annotation_path = annotation_path
        self.annotation_data = [] if annotation_data is None else annotation_data

        if new_name is None:
            file_name = os.path.split(image_paths[main_channel])[-1]
            name = os.path.splitext(file_name)[0]
            self.new_name = name
        else:
            self.new_name = new_name

    def save(self, images_dir: str = None, labels_dir: str = None):

        if images_dir is not None:
            imgs = []
            for i, image_path in enumerate(self.image_paths):

                # img = cv2.imread(image_path)
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if self.preprocessers[i] is not None:
                    img = self.preprocessers[i](img)
                imgs.append(img)
            merged_img = cv2.merge(imgs)

            # cv2.imwrite(os.path.join(images_dir, self.new_name + '.jpg'), merged_img)
            is_success, im_buf_arr = cv2.imencode(".jpg", merged_img)
            im_buf_arr.tofile(os.path.join(images_dir, self.new_name + '.jpg'))

        if labels_dir is not None:
            write_yolo_labels(os.path.join(labels_dir, self.new_name + '.txt'),
                              self.annotation_data)


class YoloDataset:
    def __len__(self):
        return len(self.labeled_images)

    def __init__(self,
                 image_dir: str = None,
                 polarization_dir: str = None,
                 annotation_data: dict = None,
                 rename_callback: Callable = None,
                 preprocessers: List[Callable] = None):

        self.labeled_images = []
        self.splits = {}

        self.add(image_dir, polarization_dir, annotation_data, rename_callback, preprocessers)

    def __getitem__(self, item):
        return self.labeled_images[item]

    def add(self,
            image_dir: str = None,
            polarization_dir: str = None,
            annotation_data: dict = None,
            rename_callback: Callable = None,
            preprocessers: List[Callable] = None):

        # Necessary data is absent - leave attribute empty and return
        if annotation_data is None or image_dir is None:
            return

        annotation_files = annotation_data['annotations'].keys()
        image_dir_files = os.listdir(image_dir)

        image_name_set = set()
        for file in image_dir_files:
            image_name_set.add(os.path.splitext(file)[0])

        for annotation_file in annotation_files:
            # if annotation_file is not correspond any image file, then skip it
            if os.path.splitext(annotation_file)[0] not in image_name_set:
                continue

            name, ext = os.path.splitext(annotation_file)
            if rename_callback is None:
                new_name = None
            else:
                new_name = rename_callback(name)

            if polarization_dir is None:
                labeled_image = DefaultLabeledImage(image_path=os.path.join(image_dir, annotation_file),
                                                    annotation_data=annotation_data['annotations'][annotation_file],
                                                    new_name=new_name)
            else:
                channel_1_img_path = os.path.join(polarization_dir, name + '_1' + ext)
                channel_2_img_path = os.path.join(polarization_dir, name + '_2' + ext)
                channel_3_img_path = os.path.join(image_dir, annotation_file)

                labeled_image = SplittedLabeledImage(image_paths=[channel_1_img_path,
                                                                  channel_2_img_path,
                                                                  channel_3_img_path],
                                                     preprocessers=preprocessers,
                                                     annotation_data=annotation_data['annotations'][annotation_file],
                                                     new_name=new_name)

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
        split_names = [name for name in os.listdir(yolo_dataset_path) if os.path.isdir(name)]

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

    def install(self, dataset_path: str):
        for split_name in self.splits.keys():
            split_idx = self.splits[split_name]

            os.makedirs(os.path.join(dataset_path, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split_name, 'labels'), exist_ok=True)

            for i in split_idx:
                images_dir = os.path.join(dataset_path, split_name, 'images')
                labels_dir = os.path.join(dataset_path, split_name, 'labels')
                print(self.labeled_images[i].image_paths[2])
                self.labeled_images[i].save(images_dir, labels_dir)




