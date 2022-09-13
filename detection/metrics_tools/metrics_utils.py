import glob
import os


import _init_paths
from BoundingBox import BoundingBox
from utils import BBType, CoordinatesType, BBFormat
from .custom_bounding_boxes import CustomBoundingBoxes

IMG_SIZE = (2448, 2048)


def get_bounding_boxes(txt_files_dir: str, bb_type: BBType) -> CustomBoundingBoxes:
    bounding_boxes = CustomBoundingBoxes()
    txt_files = glob.glob(os.path.join(txt_files_dir, "*.txt"))
    txt_files.sort()

    for txt_file in txt_files:
        img_name = os.path.split(txt_file.replace(".txt", ""))[-1]

        with open(txt_file, 'r') as f:
            lines = f.read().split('\n')

            for line in lines:
                if line == '':
                    continue

                if bb_type == BBType.GroundTruth:
                    cls_id, x, y, w, h = list(map(float, line.split(' ')))[:5]
                    bbox = BoundingBox(img_name, cls_id,
                                       x, y, w, h,
                                       CoordinatesType.Relative, IMG_SIZE,
                                       bb_type,
                                       format=BBFormat.XYWH)
                else:
                    cls_id, x, y, w, h, cls_conf = list(map(float, line.split(' ')))
                    bbox = BoundingBox(img_name, cls_id,
                                       x, y, w, h,
                                       CoordinatesType.Relative, IMG_SIZE,
                                       bb_type,
                                       cls_conf,
                                       format=BBFormat.XYWH)

                bounding_boxes.addBoundingBox(bbox)

    return bounding_boxes

