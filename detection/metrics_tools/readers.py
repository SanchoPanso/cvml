import os
import sys
import glob

from .bounding_boxes import BoundingBoxes
from .bounding_box import BoundingBox
from .metrics_utils import BBType, BBFormat, CoordinatesType

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset_tools.io_handling import read_yolo_labels

# IMG_SIZE = (2448, 2048)


def get_bounding_boxes_from_file(txt_file: str, bb_type: BBType, img_size: tuple) -> BoundingBoxes:
    img_name, ext = os.path.splitext(txt_file)
    labels = read_yolo_labels(txt_file)
    bounding_boxes = BoundingBoxes()

    for line in labels:
        if bb_type == BBType.GroundTruth:
            cls_id, x, y, w, h = line[:5]
            bbox = BoundingBox(img_name, cls_id,
                               x, y, w, h,
                               CoordinatesType.Relative, img_size,
                               bb_type,
                               format=BBFormat.XYWH)
        else:
            cls_id, x, y, w, h, cls_conf = line
            bbox = BoundingBox(img_name, cls_id,
                               x, y, w, h,
                               CoordinatesType.Relative, img_size,
                               bb_type,
                               cls_conf,
                               format=BBFormat.XYWH)

        bounding_boxes.addBoundingBox(bbox)
    return bounding_boxes


def get_bounding_boxes_from_dir(txt_files_dir: str, bb_type: BBType, img_size: tuple) -> BoundingBoxes:
    bounding_boxes = BoundingBoxes()
    txt_files = glob.glob(os.path.join(txt_files_dir, "*.txt"))
    txt_files.sort()

    for txt_file in txt_files:
        img_name = os.path.split(txt_file.replace(".txt", ""))[-1]
        bounding_boxes += get_bounding_boxes_from_file(txt_file, bb_type, img_size)

        # with open(txt_file, 'r') as f:
        #     lines = f.read().split('\n')

        #     for line in lines:
        #         if line == '':
        #             continue

        #         if bb_type == BBType.GroundTruth:
        #             cls_id, x, y, w, h = list(map(float, line.split(' ')))[:5]
        #             bbox = BoundingBox(img_name, cls_id,
        #                                x, y, w, h,
        #                                CoordinatesType.Relative, img_size,
        #                                bb_type,
        #                                format=BBFormat.XYWH)
        #         else:
        #             cls_id, x, y, w, h, cls_conf = list(map(float, line.split(' ')))
        #             bbox = BoundingBox(img_name, cls_id,
        #                                x, y, w, h,
        #                                CoordinatesType.Relative, img_size,
        #                                bb_type,
        #                                cls_conf,
        #                                format=BBFormat.XYWH)

        #         bounding_boxes.addBoundingBox(bbox)

    return bounding_boxes
