import os
from typing import List, Dict
from cvml.annotation.bounding_box import BoundingBox
from cvml.annotation.annotation import Annotation


def change_classes_by_id(annotation: Annotation, id_changes: Dict[int, int or None]) -> Annotation:
    """Change classes in annotation. Only classes, noted in id_changes, will be changed
    """
    new_annotation = Annotation()
    
    for image_name in annotation.bbox_map.keys():
        new_annotation.bbox_map[image_name] = []
        bboxes = annotation.bbox_map[image_name]
        for bbox in bboxes:
            cls_id = bbox.get_class_id()
            changed_cls_id = get_changed_class_id(cls_id, id_changes)

            if changed_cls_id is None:
                continue
            
            bbox._class_id = changed_cls_id # CHECK
            new_annotation.bbox_map[image_name].append(bbox)
    
    return new_annotation


def get_changed_class_id(class_id: int, id_changes: Dict[int, int or None]) -> int:
    if class_id in id_changes.keys():
        if id_changes[class_id] is None:
            return None
        else:
            return id_changes[class_id]
    return class_id


def change_classes_by_names(annotation: Annotation, name_changes: Dict[str, str or None]) -> Annotation:
    raise NotImplementedError


def change_classes_by_new_classes(annotation: Annotation, new_classes: List[str]) -> Annotation:
    
    new_annotation = Annotation(classes=new_classes)
    new_id_dict = {new_cls_name: i for i, new_cls_name in enumerate(new_classes)}
    
    for image_name in annotation.bbox_map.keys():
        new_annotation.bbox_map[image_name] = []
        bboxes = annotation.bbox_map[image_name]

        for bbox in bboxes:
            cls_id = bbox.get_class_id()
            cls_name = annotation.classes[cls_id]
            
            if cls_name not in new_classes:
                continue
            
            new_cls_id = new_id_dict[cls_name]
            
            new_bbox = bbox
            new_bbox._class_id = new_cls_id # Change
            new_annotation.bbox_map[image_name].append(bbox)
    
    return new_annotation


