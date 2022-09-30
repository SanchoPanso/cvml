import os
import cv2
import numpy as np
from .yolov5_detector import Yolov5Detector


class Visualiser:
    def __init__(self):
        self.gt_color = (0, 255, 0)
        self.det_color = (0, 0, 255)
        self.classes = {0: 'defect', 1: 'joint'}
        self.splits = ['train', 'test']
        self.pred_setting = {'conf': 0.1, 'count_part_x': 1, 'count_part_y': 1}
        self.skip_existing_img = False

    def create_visualization(self,
                             dataset_dir: str,
                             save_dir: dir,
                             predictor):

        for split in self.splits:
            os.makedirs(os.path.join(save_dir, split), exist_ok=True)

            images_dir = os.path.join(dataset_dir, split, 'images')
            labels_dir = os.path.join(dataset_dir, split, 'labels')

            images = os.listdir(images_dir)
            images.sort()

            for img_file in images:

                if os.path.exists(os.path.join(save_dir, img_file)) and self.skip_existing_img:
                    continue

                img_name = os.path.splitext(img_file)[0]
                print(img_name)

                # read image
                img = cv2.imread(os.path.join(images_dir, img_file))[:, :, ::-1]  # OpenCV image (BGR to RGB)

                gray = cv2.split(img)[0]
                gray = cv2.merge((gray, gray, gray))

                # draw detected bboxes
                pred = predictor(img, **self.pred_setting)
                pred_records = self.read_predictions(pred)
                for record in pred_records:
                    x, y, w, h, conf, cls_id = record
                    bbox_text = self.classes[cls_id] + ' ' + '{:.3f}'.format(conf)
                    self.draw_bbox(gray, int(x), int(y), int(w), int(h), self.det_color,
                                   bbox_text)

                # draw ground truth bboxes
                txt_file = img_name + '.txt'
                gt_file = os.path.join(labels_dir, txt_file)
                gt_records = self.read_gt_file(gt_file, img.shape)
                for record in gt_records:
                    x, y, w, h, cls_id = record
                    bbox_text = self.classes[cls_id]
                    self.draw_bbox(gray, int(x), int(y), int(w), int(h), self.gt_color,
                                   bbox_text)

                # save new image
                cv2.imwrite(os.path.join(save_dir, split, img_name + '.jpg'), gray)

    def read_predictions(self, pred: np.ndarray) -> list:
        result = []
        for i in range(pred.shape[0]):
            x, y, w, h, conf, cls_id = pred[i]
            result.append((x, y, w, h, conf, cls_id))
        return result

    def read_gt_file(self, path: str, img_shape: tuple) -> list:
        result = []
        with open(path) as f:
            lines = f.read().strip().split('\n')

        for line in lines:
            if line == '':
                continue
            cls_id, xc, yc, w, h = map(float, line.split())
            xc *= img_shape[1]
            yc *= img_shape[0]
            w *= img_shape[1]
            h *= img_shape[0]
            x = xc - w / 2
            y = yc - h / 2
            result.append((x, y, w, h, cls_id))

        return result

    def draw_bbox(self, img: np.ndarray,
                  x: int, y: int, w: int, h: int,
                  color: tuple,
                  description: str = ''):
        cv2.rectangle(img,
                      (x, y),
                      (x + w, y + h),
                      color, 3)
        cv2.putText(img,
                    description,
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

