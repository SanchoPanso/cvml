import os
import json
from typing import Callable, List, Dict

from data_science_tools.core.bounding_box import BoundingBox, CoordinatesType
from data_science_tools.detection.dataset.io_handling import read_yolo_labels


class Annotation:
    """
    Representation of the detection annotation of some dataset.
    """
    def __init__(self, 
                 classes: List[str] = None, 
                 bounding_boxes: Dict[str, List[BoundingBox]] = None):
        """
        :classes: list of class names
        :bounding_boxes: dict with keys - image names and values - list of bounding boxes on this image
        """
        
        self.classes = [] if classes is None else classes
        self.bounding_boxes = {} if bounding_boxes is None else bounding_boxes


class AnnotationConverter:
    def __init__(self):
        pass

    def read_coco(self, path: str) -> Annotation:
        """
        :path: absolute path to json file with coco annotation
        :return: annotation extracted from json file  
        """
        with open(path) as f:
            input_data = json.load(f)

        # Prepare special dicts of data
        classes = self._get_classes_from_coco(input_data)
        images = self._get_images_from_coco(input_data)
        bboxes = self._get_bboxes_from_coco(input_data)

        bb_dict = {}

        # For each image in coco create an empty list of bounding boxes
        for key in images.keys():
            file_name = images[key]['file_name']
            name, ext = os.path.splitext(file_name)
            bb_dict[name] = []
        
        # Each bbox in coco add to an appropriate list of bboxes
        for bbox_id in bboxes.keys():
            bbox = self.get_bounding_box_from_coco_data(bbox_id, classes, images, bboxes)
            name = bbox.get_image_name()
            bb_dict[name].append(bbox)

        # Create list of class names
        classes_list = ['' for key in classes.keys()]
        for key in classes.keys():
            cls_num = classes[key]['cls_num']
            classes_list[cls_num] = classes[key]['cls_name']

        # Create instance of Annotation and return
        annotation = Annotation(classes_list, bb_dict)
        return annotation

    def write_coco(self, annotation: Annotation, path: str):
        """
        :annotation: annotation to convert
        :path: absolute path for saving coco json  
        """
        
        # Get special dicts of coco-format from the specific annotation
        categories = self._get_categories_from_annotation(annotation)
        images = self._get_images_from_annotation(annotation)
        annotations = self._get_bboxes_from_annotation(annotation)

        # Create default coco data
        license = [{"name": "", "id": 0, "url": ""}]
        info = {"contributor": "", 
                "date_created": "", 
                "description": "", 
                "url": "", 
                "version": "", 
                "year": ""}
        
        # Create coco dict and save
        coco = {
            'license': license,
            'info': info,
            'categories': categories,
            'images': images,
            'annotations': annotations, 
        }

        with open(path, 'w') as f:
            json.dump(coco, f)
    
    def read_bboxes(self, bboxes: List[BoundingBox], classes: List[str]) -> Annotation:
        """
        :bboxes: list of bounding boxes to convert into an annotation
        :classes: list of class names
        :return: converted annotation
        """

        bb_dict = {}
        for bb in bboxes:
            image_name = bb.get_image_name()
            if image_name in bb_dict.keys():
                bb_dict[image_name].append(bb)
            else:
                bb_dict[image_name] = [bb]

        annotation = Annotation(classes, bb_dict)
        return annotation
    
    def read_yolo(self, path: str, classes: List[str] = None, data_yaml_path: str = None) -> Annotation:
        """
        :path: absolute path to labels dir with txt-files of yolo annotation
        :classes: list of class names
        :data_yaml_path: path to data.yaml in yolo dataset
        :return: annotation extracted from these files  
        """
        max_cls_id = -1
        txt_files = os.listdir(path) 

        bb_dict = {}
        # For each image in coco create an empty list of bounding boxes
        for file in txt_files:
            name, ext = os.path.splitext(file)
            bboxes = read_yolo_labels(os.path.join(path, file))

            for bb in bboxes:
                max_cls_id = max(max_cls_id, bb.get_class_id())
            bb_dict[name] = bboxes
        
        if classes is not None:
            pass # TODO
        else:
            classes = [str(i) for i in range(int(max_cls_id + 1))]
        
        annotation = Annotation(classes, bb_dict)
        return annotation

    def _get_classes_from_coco(self, coco_dict: dict) -> dict:
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

    def _get_images_from_coco(self, coco_dict: dict) -> dict:
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

    def _get_bboxes_from_coco(self, coco_dict: dict) -> dict:
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

    def _get_categories_from_annotation(self, annotation: Annotation) -> dict:
        categories = []
        for i, cls in enumerate(annotation.classes):
            category = {
                "id": i + 1, 
                "name": cls, 
                "supercategory": ""
            }
            categories.append(category)
        return categories

    def _get_images_from_annotation(self, annotation: Annotation) -> dict:
        
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
                "file_name": image_name + '.jpg', # REDO
                "license": 0, 
                "flickr_url": "", 
                "coco_url": "", 
                "date_captured": 0
            }
            images.append(image)
        return images
    
    def _get_bboxes_from_annotation(self, annotation: Annotation) -> dict:
        coco_annotations = []
        bbox_id = 1
        for i, image_name in enumerate(annotation.bounding_boxes.keys()):
            if len(annotation.bounding_boxes[image_name]) == 0:
                continue
            img_id = i + 1
            for bbox in annotation.bounding_boxes[image_name]:
                x, y, w, h = bbox.get_absolute_bounding_box()
                cls_id = bbox.get_class_id()
                segmentation = bbox.get_segmentation()
                coco_annotation = {
                    "id": bbox_id, 
                    "image_id": img_id, 
                    "category_id": cls_id + 1, 
                    "segmentation": segmentation, 
                    "area": w * h, 
                    "bbox": [x, y, w, h], 
                    "iscrowd": 0, 
                    "attributes*": {"occluded": False, "rotation": 0.0}
                }
                bbox_id += 1
                coco_annotations.append(coco_annotation)
        return coco_annotations

    




