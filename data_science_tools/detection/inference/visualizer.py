import os
import cv2
import numpy as np
from typing import List
from .yolov5_detector import Yolov5Detector
from data_science_tools.core.bounding_box import BoundingBox, BBFormat, BBType


class Visualiser:
    def __init__(self):
        pass
        # self.gt_color = (0, 255, 0)
        # self.det_color = (0, 0, 255)
        # self.classes = {0: 'defect', 1: 'joint'}
        # self.splits = ['train', 'test']
        # self.pred_setting = {'conf': 0.1, 'count_part_x': 1, 'count_part_y': 1}
        # self.skip_existing_img = False

    # def create_visualization(self,
    #                          dataset_dir: str,
    #                          save_dir: dir,
    #                          predictor):

    #     for split in self.splits:
    #         os.makedirs(os.path.join(save_dir, split), exist_ok=True)

    #         images_dir = os.path.join(dataset_dir, split, 'images')
    #         labels_dir = os.path.join(dataset_dir, split, 'labels')

    #         images = os.listdir(images_dir)
    #         images.sort()

    #         for img_file in images:

    #             if os.path.exists(os.path.join(save_dir, img_file)) and self.skip_existing_img:
    #                 continue

    #             img_name = os.path.splitext(img_file)[0]
    #             print(img_name)

    #             # read image
    #             img = cv2.imread(os.path.join(images_dir, img_file))[:, :, ::-1]  # OpenCV image (BGR to RGB)

    #             gray = cv2.split(img)[0]
    #             gray = cv2.merge((gray, gray, gray))

    #             # draw detected bboxes
    #             pred = predictor(img, **self.pred_setting)
    #             pred_records = self.read_predictions(pred)
    #             for record in pred_records:
    #                 x, y, w, h, conf, cls_id = record
    #                 bbox_text = self.classes[cls_id] + ' ' + '{:.3f}'.format(conf)
    #                 self.draw_bbox(gray, int(x), int(y), int(w), int(h), self.det_color,
    #                                bbox_text)

    #             # draw ground truth bboxes
    #             txt_file = img_name + '.txt'
    #             gt_file = os.path.join(labels_dir, txt_file)
    #             gt_records = self.read_gt_file(gt_file, img.shape)
    #             for record in gt_records:
    #                 x, y, w, h, cls_id = record
    #                 bbox_text = self.classes[cls_id]
    #                 self.draw_bbox(gray, int(x), int(y), int(w), int(h), self.gt_color,
    #                                bbox_text)

    #             # save new image
    #             cv2.imwrite(os.path.join(save_dir, split, img_name + '.jpg'), gray)

    # def read_predictions(self, pred: np.ndarray) -> list:
    #     result = []
    #     for i in range(pred.shape[0]):
    #         x, y, w, h, conf, cls_id = pred[i]
    #         result.append((x, y, w, h, conf, cls_id))
    #     return result

    # def read_gt_file(self, path: str, img_shape: tuple) -> list:
    #     result = []
    #     with open(path) as f:
    #         lines = f.read().strip().split('\n')

    #     for line in lines:
    #         if line == '':
    #             continue
    #         cls_id, xc, yc, w, h = map(float, line.split())
    #         xc *= img_shape[1]
    #         yc *= img_shape[0]
    #         w *= img_shape[1]
    #         h *= img_shape[0]
    #         x = xc - w / 2
    #         y = yc - h / 2
    #         result.append((x, y, w, h, cls_id))

    #     return result

    def draw_bbox(self, 
                  img: np.ndarray,
                  bbox: BoundingBox,
                  classes: List[str],
                  color: tuple,
                  thickness=2):

        r, g, b = color

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1

        x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), thickness)

        cls_id = bbox.get_class_id()
        if bbox.get_bb_type() == BBType.GroundTruth:
            label = classes[cls_id]
        else:
            label = '{0} {1:3f}'.format(classes[cls_id])

        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
