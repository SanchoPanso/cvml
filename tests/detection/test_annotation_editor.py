import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from cvml.annotation.bounding_box import BoundingBox
from cvml import Annotation, write_coco, write_yolo, change_classes_by_id


def test_change_classes_by_id():
    annotation = Annotation()
    annotation.classes = ['0', '1', '2', '3', '4']
    annotation.bbox_map['image_name'] = [
        BoundingBox(0, 25, 25, 25, 25),
        BoundingBox(1, 25, 25, 25, 25),
        BoundingBox(2, 25, 25, 25, 25),
        BoundingBox(3, 25, 25, 25, 25),
    ]
    
    id_changes = {0: None, 1: 4}
    new_annotation = change_classes_by_id(annotation, id_changes)

    assert new_annotation.bbox_map['image_name'][0].get_class_id() == 4
    assert new_annotation.bbox_map['image_name'][1].get_class_id() == 2
    assert new_annotation.bbox_map['image_name'][2].get_class_id() == 3


