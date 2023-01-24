import os
import json


class Extractor:
    def __init__(self):
        pass

    def extract(self, input_annotation: str) -> dict:
        with open(input_annotation) as f:
            input_data = json.load(f)

        classes = self.get_classes(input_data)
        images = self.get_images(input_data)
        bboxes = self.get_bboxes(input_data)

        annotations = {}

        for key in images.keys():
            file_name = images[key]['file_name']
            annotations[file_name] = []

        for i, bbox_id in enumerate(bboxes.keys()):

            record = self.get_bbox_record(bbox_id, classes, images, bboxes)
            file_name, cls_num, x, y, w, h = record
            annotations[file_name].append([cls_num, x, y, w, h])

        classes_list = [0 for key in classes.keys()]
        for key in classes.keys():
            cls_num = classes[key]['cls_num']
            classes_list[cls_num] = classes[key]['cls_name']

        result = {
            'classes': classes_list,
            'annotations': annotations,
        }
        return result

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

        x, y, w, h = bbox
        x = (x + w/2) / width
        y = (y + h/2) / height
        w /= width
        h /= height

        return file_name, cls_num, x, y, w, h

