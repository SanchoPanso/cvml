import os
import json
from typing import Callable


class Extractor:
    def __init__(self):
        pass

    def __call__(self, path: str, annotation_type: str = 'coco') -> dict:
        """
        Extract detection info from coco-annotation into yolov5-like form
        :param path: path to json-file with coco-annotation

        :return: dictionary with detection info in from like this:
        {'classes': ['cls1', 'cls2', ...],
         'annotations':{'1.png': [0, 0.1, 0.2, 0.05, 0.05],
                        '2.png': [0, 0.1, 0.2, 0.05, 0.05],
                        ...
                        }
         }
         Key 'classes' - list of cls names
         Key 'annotations' - dict with keys - img name, value - bbox in form [cls_id, x_c, y_c, w, h]
        """

        extract_function = self.get_extract_function(annotation_type)
        annotation_data = extract_function(path)

        return annotation_data

    def get_extract_function(self, annotation_type: str) -> Callable:
        if annotation_type == 'coco':
            return self.extract_coco
        elif annotation_type == 'yolo':
            return self.extract_yolo
        else:
            raise ValueError("Wrong annotation type")

    def extract_coco(self, path: str) -> dict:
        with open(path) as f:
            input_data = json.load(f)

        classes = self.get_classes(input_data)
        images = self.get_images(input_data)
        bboxes = self.get_bboxes(input_data)

        annotations = {}

        for key in images.keys():
            file_name = images[key]['file_name']
            name, ext = os.path.splitext(file_name)
            annotations[name] = []

        for i, bbox_id in enumerate(bboxes.keys()):
            record = self.get_bbox_record(bbox_id, classes, images, bboxes)
            name, cls_num, x, y, w, h = record
            annotations[name].append([cls_num, x, y, w, h])

        classes_list = [0 for key in classes.keys()]
        for key in classes.keys():
            cls_num = classes[key]['cls_num']
            classes_list[cls_num] = classes[key]['cls_name']

        result = {
            'classes': classes_list,
            'annotations': annotations,
        }
        return result

    def extract_yolo(self, path: str) -> dict:
        return {}

    def get_classes(self, input_data: dict) -> dict:
        input_categories = input_data['categories']
        result = {}
        for cls_num in range(len(input_categories)):
            cls_id = input_categories[cls_num]['id']
            cls_name = input_categories[cls_num]['name']
            result[cls_id] = {
                'cls_num': cls_num,
                'cls_name': cls_name,
            }
        return result

    def get_images(self, input_data: dict) -> dict:
        input_images = input_data['images']
        result = {}
        for images_num in range(len(input_images)):
            image_id = input_images[images_num]['id']
            width = input_images[images_num]['width']
            height = input_images[images_num]['height']
            file_name = input_images[images_num]['file_name']
            result[image_id] = {
                'image_id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name,
            }
        return result

    def get_bboxes(self, input_data: dict) -> dict:
        input_annotations = input_data['annotations']
        result = {}
        for bbox_num in range(len(input_annotations)):
            bbox_id = input_annotations[bbox_num]['id']
            image_id = input_annotations[bbox_num]['image_id']
            cls_id = input_annotations[bbox_num]['category_id']
            bbox = input_annotations[bbox_num]['bbox']
            result[bbox_id] = {
                'bbox_num': bbox_num,
                'image_id': image_id,
                'cls_id': cls_id,
                'bbox': bbox,
            }
        return result

    def get_bbox_record(self, bbox_id: str, classes: dict, images: dict, bboxes: dict) -> tuple:
        bbox = bboxes[bbox_id]['bbox']
        image_id = bboxes[bbox_id]['image_id']
        cls_id = bboxes[bbox_id]['cls_id']

        cls_num = classes[cls_id]['cls_num']

        width = images[image_id]['width']
        height = images[image_id]['height']
        file_name = images[image_id]['file_name']

        name, ext = os.path.splitext(file_name)

        x, y, w, h = bbox
        x = (x + w/2) / width
        y = (y + h/2) / height
        w /= width
        h /= height

        return name, cls_num, x, y, w, h

