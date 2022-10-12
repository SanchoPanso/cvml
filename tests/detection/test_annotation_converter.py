import pytest
import os
import sys
import json

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)

from cvml.core.bounding_box import BBType, BoundingBox, CoordinatesType, BBFormat
from cvml.detection.dataset.annotation_converter import AnnotationConverter, Annotation


def test_read_coco():
    converter = AnnotationConverter()
    coco_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_coco.json')
    annot = converter.read_coco(coco_path)
    bbox_map = annot.bbox_map
    images = list(bbox_map.keys())

    assert annot.classes == ['comet', 'other']
    assert images == ['1', '10']

    assert len(bbox_map['1']) == 1
    assert len(bbox_map['10']) == 1
    
    assert bbox_map['1'][0].get_class_id() == 0
    assert bbox_map['1'][0].get_segmentation() == [[1200, 500, 1260, 500, 1200, 1050]]
    assert bbox_map['1'][0].get_absolute_bounding_box() == (1200, 500, 60, 550)
    assert bbox_map['1'][0].get_image_size() == (2448, 2048)
    assert bbox_map['1'][0].get_confidence() == 1.0
    assert bbox_map['1'][0].get_bb_type() == BBType.GroundTruth

    assert bbox_map['10'][0].get_class_id() == 1
    assert bbox_map['10'][0].get_segmentation() == {"size": [2448, 2048], "counts": [0, 1]}
    assert bbox_map['10'][0].get_absolute_bounding_box() == (560, 820, 60, 130)
    assert bbox_map['10'][0].get_image_size() == (2448, 2048)
    assert bbox_map['10'][0].get_confidence() == 1.0
    assert bbox_map['10'][0].get_bb_type() == BBType.GroundTruth


def test_write_coco():

    bbox_1 = BoundingBox(0, 1200, 500, 60, 550,
                         class_confidence=1.0,
                         image_name='1',
                         type_coordinates=CoordinatesType.Absolute,
                         img_size=(2448, 2048),
                         bb_type=BBType.GroundTruth,
                         format=BBFormat.XYWH,
                         segmentation=[[1200, 500, 1260, 500, 1200, 1050]])
    bbox_2 = BoundingBox(1, 560, 820, 60, 130,
                         class_confidence=1.0,
                         image_name='10',
                         type_coordinates=CoordinatesType.Absolute,
                         img_size=(2448, 2048),
                         bb_type=BBType.GroundTruth,
                         format=BBFormat.XYWH,
                         segmentation={"size": [2448, 2048], "counts": [0, 1]})
    
    classes = ['comet', 'other']
    bbox_map = {'1': [bbox_1], '10': [bbox_2]}
    annot = Annotation(classes, bbox_map)
    converter = AnnotationConverter()
    result_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_coco_2.json')
    converter.write_coco(annot, result_path, image_ext='.png')

    coco_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_coco.json')
    with open(result_path) as f:
        got_dict = json.load(f)
    with open(coco_path) as f:
        expected_dict = json.load(f)
    
    assert got_dict['licenses'] == expected_dict['licenses']
    assert got_dict['info'] == expected_dict['info']
    assert got_dict['images'] == expected_dict['images']
    assert got_dict['annotations'] == expected_dict['annotations']





