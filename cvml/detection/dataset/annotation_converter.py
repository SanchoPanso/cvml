import os
import json
from typing import Callable, List, Dict

from cvml.core.bounding_box import BoundingBox, CoordinatesType
from cvml.detection.dataset.annotation import Annotation


class AnnotationConverter:
    def __init__(self):
        pass
    
    @classmethod
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

    @classmethod
    def write_coco(self, annotation: Annotation, path: str, image_ext: str = '.jpg'):
        """
        :annotation: annotation to convert
        :path: absolute path for saving coco json  
        :image_ext: file extension, under which images will be saved
        """
        
        # Get special dicts of coco-format from the specific annotation
        categories = self._get_categories_from_annotation(annotation)
        images = self._get_images_from_annotation(annotation, image_ext)
        annotations = self._get_bboxes_from_annotation(annotation)

        # Create default coco data
        licenses = [{"name": "", "id": 0, "url": ""}]
        info = {"contributor": "", 
                "date_created": "", 
                "description": "", 
                "url": "", 
                "version": "", 
                "year": ""}
        
        # Create coco dict and save
        coco = {
            'licenses': licenses,
            'info': info,
            'categories': categories,
            'images': images,
            'annotations': annotations, 
        }

        with open(path, 'w') as f:
            json.dump(coco, f)
    
    @classmethod
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
    
    @classmethod
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
            segmentation = annotations[bbox_num]['segmentation']
            result[bbox_id] = {
                'bbox_num': bbox_num,
                'image_id': image_id,
                'cls_id': cls_id,
                'bbox': bbox,
                'segmentation': segmentation,
            }
        return result

    def get_bounding_box_from_coco_data(self, bbox_id: str, classes: dict, images: dict, bboxes: dict) -> BoundingBox:
        bbox = bboxes[bbox_id]['bbox']
        image_id = bboxes[bbox_id]['image_id']
        cls_id = bboxes[bbox_id]['cls_id']
        segm = bboxes[bbox_id]['segmentation']

        cls_num = classes[cls_id]['cls_num']

        width = images[image_id]['width']
        height = images[image_id]['height']
        file_name = images[image_id]['file_name']

        name, ext = os.path.splitext(file_name)

        x, y, w, h = bbox

        bounding_box = BoundingBox(cls_num, x, y, w, h, 
                                   image_name=name, 
                                   type_coordinates=CoordinatesType.Absolute, 
                                   img_size=(width, height),
                                   class_confidence=1.0,
                                   segmentation=segm)

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

    def _get_images_from_annotation(self, annotation: Annotation, image_ext: str) -> dict:
        
        images = []
        img_id_dict = {}
        for i, image_name in enumerate(annotation.bbox_map.keys()):
            if len(annotation.bbox_map[image_name]) == 0:
                continue
            img_id = i + 1
            img_id_dict[image_name] = img_id
            image = {
                "id": img_id, 
                "width": 2448, 
                "height": 2048, 
                "file_name": image_name + image_ext,
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
        for i, image_name in enumerate(annotation.bbox_map.keys()):
            if len(annotation.bbox_map[image_name]) == 0:
                continue
            img_id = i + 1
            for bbox in annotation.bbox_map[image_name]:
                x, y, w, h = bbox.get_absolute_bounding_box()
                cls_id = bbox.get_class_id()
                segmentation = bbox.get_segmentation()
                coco_annotation = {
                    "id": bbox_id, 
                    "image_id": img_id, 
                    "category_id": cls_id + 1, 
                    "segmentation": segmentation, 
                    "area": float(w * h), 
                    "bbox": list(map(float, [x, y, w, h])), 
                    "iscrowd": 0, 
                    "attributes": {"occluded": False, "rotation": 0.0}
                }
                bbox_id += 1
                coco_annotations.append(coco_annotation)
        return coco_annotations


def read_yolo_labels(path: str, img_size: tuple) -> List[BoundingBox]:

    image_name = os.path.splitext(os.path.split(path)[-1])[0]
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.read().split('\n')
        bboxes = []
        for row in rows:
            if row == '':
                continue
            row_data = list(map(float, row.split(' ')))
            if len(row_data) == 5:
                cls_id, xc, yc, w, h = row_data
                # x = xc - w / 2
                # y = yc - h / 2
                cls_conf = None
            else:  # 6
                cls_id, xc, yc, w, h, cls_conf = row_data
                # x = xc - w / 2
                # y = yc - h / 2

            bbox = BoundingBox(cls_id, xc, yc, w, h, cls_conf, 
                               type_coordinates=CoordinatesType.Relative,
                               image_name=image_name,
                               img_size=img_size)
            bboxes.append(bbox)
    return bboxes


def write_yolo_labels(path: str, bboxes: List[BoundingBox]):
    with open(path, 'w') as f:
        for bbox in bboxes:
            cls_id = bbox.get_class_id()
            xc, yc, w, h = bbox.get_relative_bounding_box()
            line = f"{cls_id} {xc} {yc} {w} {h}\n"
            f.write(line)
    




