import os
from typing import List
from cvml.core.bounding_box import BoundingBox, CoordinatesType


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


# def read_yolo_labels(path: str) -> list:
#     with open(path, 'r', encoding='utf-8') as f:
#         rows = f.read().split('\n')
#         lines = []
#         for row in rows:
#             if row == '':
#                 continue
#             lines.append(list(map(float, row.split(' '))))
#     return lines


# def write_yolo_labels(path: str, lines: list):
#     with open(path, 'w') as f:
#         for line in lines:
#             f.write(' '.join(list(map(str, line))) + '\n')

