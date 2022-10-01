import os
import json
from typing import Callable, List, Dict

from data_science_tools.core.bounding_box import BoundingBox, CoordinatesType


class Annotation:
    def __init__(self, 
                 classes: List[str] = None, 
                 bounding_boxes: Dict[str, List[BoundingBox]] = None):
        
        self.classes = [] if classes is None else classes
        self.bounding_boxes = {} if bounding_boxes is None else bounding_boxes


class AnnotationConverter:
    def __init__(self):
        pass

    def read_coco(self, path: str) -> Annotation:
        with open(path) as f:
            input_data = json.load(f)

        classes = self.get_classes_from_coco(input_data)
        images = self.get_images_from_coco(input_data)
        bboxes = self.get_bboxes_from_coco(input_data)

        bb_dict = {}

        for key in images.keys():
            file_name = images[key]['file_name']
            name, ext = os.path.splitext(file_name)
            bb_dict[name] = []

        for i, bbox_id in enumerate(bboxes.keys()):
            bbox = self.get_bounding_box_from_coco_data(bbox_id, classes, images, bboxes)
            name = bbox.get_image_name()
            bb_dict[name].append(bbox)

        classes_list = ['' for key in classes.keys()]
        for key in classes.keys():
            cls_num = classes[key]['cls_num']
            classes_list[cls_num] = classes[key]['cls_name']

        annotation = Annotation(classes_list, bb_dict)
        return annotation

    def read_yolo(self, path: str) -> Annotation:
        return Annotation()

    def write_coco(self, annotation: Annotation, path: str):
        
        categories = self.get_categories_from_annotation(annotation)
        images = self.get_images_from_annotation(annotation)
        annotations = self.get_bboxes_from_annotation(annotation)

        license = [{"name": "", "id": 0, "url": ""}]
        info = {"contributor": "", 
                "date_created": "", 
                "description": "", 
                "url": "", 
                "version": "", 
                "year": ""}
        
        coco = {
            'license': license,
            'info': info,
            'categories': categories,
            'images': images,
            'annotations': annotations, 
        }

        with open(path) as f:
            json.dump(coco, f)


    def get_classes_from_coco(self, coco_dict: dict) -> dict:
        categories = coco_dict['categories']
        result = {}
        for cls_num in range(len(categories)):
            cls_id = categories[cls_num]['id']
            cls_name = categories[cls_num]['name']
            result[cls_id] = {
                'cls_num': cls_num,
                'cls_name': cls_name,
            }
        return result

    def get_images_from_coco(self, coco_dict: dict) -> dict:
        images = coco_dict['images']
        result = {}
        for images_num in range(len(images)):
            image_id = images[images_num]['id']
            width = images[images_num]['width']
            height = images[images_num]['height']
            file_name = images[images_num]['file_name']
            result[image_id] = {
                'image_id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name,
            }
        return result

    def get_bboxes_from_coco(self, coco_dict: dict) -> dict:
        annotations = coco_dict['annotations']
        result = {}
        for bbox_num in range(len(annotations)):
            bbox_id = annotations[bbox_num]['id']
            image_id = annotations[bbox_num]['image_id']
            cls_id = annotations[bbox_num]['category_id']
            bbox = annotations[bbox_num]['bbox']
            result[bbox_id] = {
                'bbox_num': bbox_num,
                'image_id': image_id,
                'cls_id': cls_id,
                'bbox': bbox,
            }
        return result

    def get_bounding_box_from_coco_data(self, bbox_id: str, classes: dict, images: dict, bboxes: dict) -> BoundingBox:
        bbox = bboxes[bbox_id]['bbox']
        image_id = bboxes[bbox_id]['image_id']
        cls_id = bboxes[bbox_id]['cls_id']

        cls_num = classes[cls_id]['cls_num']

        width = images[image_id]['width']
        height = images[image_id]['height']
        file_name = images[image_id]['file_name']

        name, ext = os.path.splitext(file_name)

        x, y, w, h = bbox

        bounding_box = BoundingBox(cls_num, x, y, w, h, 
                                   image_name=name, 
                                   type_coordinates=CoordinatesType.Absolute, 
                                   img_size=(width, height))

        return bounding_box

    def get_categories_from_annotation(annotation: Annotation) -> dict:
        categories = []
        for i, cls in enumerate(annotation.classes):
            category = {
                "id": i + 1, 
                "name": cls, 
                "supercategory": ""
            }
            categories.append(category)
        return categories

    def get_images_from_annotation(annotation: Annotation) -> dict:
        
        images = []
        img_id_dict = {}
        for i, image_name in enumerate(annotation.bounding_boxes.keys()):
            if len(annotation.bounding_boxes[image_name]) == 0:
                continue
            img_id = i + 1
            img_id_dict[image_name] = img_id
            image = {
                "id": img_id, 
                "width": 2448, 
                "height": 2048, 
                "file_name": image_name, # CHECK 
                "license": 0, 
                "flickr_url": "", 
                "coco_url": "", 
                "date_captured": 0
            }
            images.append(image)
        return images
    
    def get_bboxes_from_annotation(annotation: Annotation) -> dict:
        annotations = []
        bbox_id = 1
        for i, image_name in enumerate(annotation.bounding_boxes.keys()):
            if len(annotation.bounding_boxes[image_name]) == 0:
                continue
            img_id = i + 1
            for bbox in annotation.bounding_boxes[image_name]:
                x, y, w, h = bbox.get_absolute_bounding_box()
                cls_id = bbox.get_class_id()
                annotation = {
                    "id": bbox_id, 
                    "image_id": img_id, 
                    "category_id": cls_id + 1, 
                    "segmentation": [], 
                    "area": w * h, 
                    "bbox": [x, y, w, h], 
                    "iscrowd": 0, 
                    "attributes*": {"occluded": False, "rotation": 0.0}
                }
                bbox_id += 1
                annotations.append(annotation)

    




