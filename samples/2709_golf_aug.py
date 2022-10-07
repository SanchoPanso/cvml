"""
Image augmentation functions
"""
import os
import sys
import logging
import cv2
import numpy as np
import math
import random
import torch
import time
import albumentations as A
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.detection.inference_tools.yolov5_detector import Yolov5Detector


class GOLF_Albumentations:
    def __init__(self):
        self.dir_coco_obj = r"E:\PythonProjects\AnnotationConverter\datasets\defects_segment_27092022"
        self.coco_class_names = ['comet']
        self.class_names = ['comet']

        self.coco_conformity = {
            'comet': 0,
        }

        self.class_names_id = {class_name: class_id for class_id, class_name in enumerate(self.class_names)}

        self.img_coco_obj = {}
        for class_obj in self.coco_class_names:
            dir_class_coco_obj = os.path.join(self.dir_coco_obj, class_obj)
            if not os.path.exists(dir_class_coco_obj):
                self.img_coco_obj.update({class_obj: None})
                continue
            list_img_path = [os.path.join(dir_class_coco_obj, fname) for fname in os.listdir(dir_class_coco_obj)]
            print("generate object from coco_dataset")
            print(f"{class_obj}: count images {len(list_img_path)}")
            self.img_coco_obj.update({class_obj: list_img_path.copy()})

    def __call__(self, im, labels, p=1.0, num_obj=10, **kwargs):
        height, width = im.shape[:2]
        if random.random() > p:
            return im, labels

        # random insert image coco objects in original image
        for idx in range(num_obj):
            class_coco_name = self.select_obj()
            img_obj = self.select_image_obj(class_coco_name)
            #try:
            if img_obj is None:
                continue
            im, bbox = self.random_paste_img_to_img(im, img_obj, labels_yolo=labels)
            if bbox is None:
                continue
            class_id = self.coco_conformity.get(class_coco_name)
            x_c, y_c, w_bbox, h_bbox = self.xyxy_to_xcycwh(bbox)
            x_c, y_c = x_c / width, y_c / height
            w_bbox, h_bbox = w_bbox / width, h_bbox / height
            yolo_bbox = np.array([[class_id, x_c, y_c, w_bbox, h_bbox]])
            yolo_bbox = self.fix_bbox(yolo_bbox)
            labels = np.append(labels, yolo_bbox, axis=0)
            # except Exception as e:
            #     print("error GOLF_Albumentations", e)
            #     continue

        return im, labels

    def random_paste_img_to_img(self, img, img_obj, labels_yolo):

        height, width = img.shape[0:2]
        h_obj, w_obj = img_obj.shape[0:2]

        x_tl = (width - w_obj) // 2
        x_br = (width + w_obj) // 2
        y_tl = (height - h_obj) // 2
        y_br = (height + h_obj) // 2
        bbox = [x_tl, y_tl, x_br, y_br]

        if w_obj > width or h_obj > height:
            return img, None

        iou_max = 0
        warp_mat = np.zeros((2, 3), dtype=np.float64)
        new_bbox = None
        for sample_id in range(50):
            warp_mat = self.get_random_perspective_transform(img.shape)
            new_bbox = self.warp_bbox(bbox, warp_mat)
            iou_max = self.get_iou_max(img, new_bbox, labels_yolo)
            if iou_max <= 0.3:
                break
        if iou_max > 0.3:
            return img, None

        # place img_obj on black background with size of original img
        img_obj_on_black = np.zeros(img.shape, dtype=img.dtype)
        img_obj_on_black[y_tl:y_br, x_tl:x_br] = img_obj

        # mask = self.get_mask(bbox, img.shape)

        #flip
        flip_coef = random.randint(-1, 1)
        if flip_coef != -1:
            img_obj_on_black = cv2.flip(img_obj_on_black, flip_coef)

        new_img_obj = cv2.warpPerspective(img_obj_on_black, warp_mat, (width, height))

        lower = np.array([0, 0, 35], dtype="uint8")
        upper = np.array([255, 255, 255], dtype="uint8")
        new_img_obj_hsv = cv2.cvtColor(new_img_obj, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(new_img_obj_hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        # new_mask = cv2.warpPerspective(mask, warp_mat, (width, height))

        opacity = random.uniform(0.5, 1.0)
        # final_img = self.merge_images_weighted(img, new_img_obj, new_mask, weight=opacity)

        img_without_obj = cv2.bitwise_and(img, img, mask=mask_inv)
        img_obj_placeholder = cv2.bitwise_and(img, img, mask=mask)
        final_img = cv2.addWeighted(img_obj_placeholder, 1 - opacity, new_img_obj, opacity, 0)
        final_img = cv2.addWeighted(final_img, 1, img_without_obj, 1, 0)

        # final_img = cv2.addWeighted(img, 1, new_img_obj, 1, 0)
        return final_img, new_bbox

    def fix_bbox(self, yolo_bbox):
        cls_id, xc, yc, w, h = yolo_bbox[0]

        xc = min(1, max(0, xc))
        yc = min(1, max(0, yc))
        w = min(2 * xc, 2 - 2 * xc, w)
        h = min(2 * yc, 2 - 2 * yc, h)

        yolo_bbox[0][1] = xc
        yolo_bbox[0][2] = yc
        yolo_bbox[0][3] = w
        yolo_bbox[0][4] = h

        return yolo_bbox

    def get_mask(self, bbox: list, img_shape: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    def select_image_obj(self, class_name):
        if self.img_coco_obj.get(class_name) is None:
            return None
        path_img = np.random.choice(self.img_coco_obj.get(class_name))
        try:
            img_obj = cv2.imread(path_img)
            img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
            return img_obj
        except Exception as e:
            return None

    def select_obj(self):
        while True:
            class_name = np.random.choice(list(self.coco_conformity.keys()))
            if (self.coco_conformity.get(class_name) is not None) and \
               (self.img_coco_obj.get(class_name) is not None):
                break
        return class_name

    def random_resize_img(self, img, min_size=(100,100), max_size=(200,200)):
        """
        :param
        """
        w_max, h_max  = max_size
        w_min, h_min  = min_size
        h_obj, w_obj = img.shape[:2]
        if w_obj > h_obj:
            new_w_obj = np.random.randint(low=w_min, high=w_max)
            r = new_w_obj / w_obj
            new_h_obj = int(r*h_obj)
        else:
            new_h_obj = np.random.randint(low=h_min, high=h_max)
            r = new_h_obj / h_obj
            new_w_obj = int(r * w_obj)
        img = cv2.resize(img, (new_w_obj, new_h_obj))
        return img

    def merge_images(self, back_img, top_img):
        """
        merge two image with same size
        :param back_img: background image
        :param top_img: insertion image over background  (black color = transparent color)
        """
        tmp = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 15, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(top_img)
        rgba = [b, g, r, alpha]
        rgba = cv2.merge(rgba, 4)
        alpha_channel = rgba[:, :, 3] / 255
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        
        overlay_coef = random.random()
        overlay_coef = overlay_coef if overlay_coef > 0.8 else 0.8
        top_img_weight = cv2.addWeighted(back_img, 1, top_img, overlay_coef, 0.0)
        composite = back_img * (1 - alpha_mask) + top_img_weight * alpha_mask
        
        #composite = back_img * (1 - alpha_mask) + top_img * alpha_mask
        composite = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)
        return composite

    def get_iou_max(self, img, bboxA, labels_yolo):
        """
        Get maximum IoU bbox object in labels_yolo
        :param img: original image
        :param bboxA: bbox object
        :param labels_yolo: all coordinates bboxes in image
                           (format yolo - class_id, x_c, y_c, w, h)
        """
        height, width = img.shape[:2]
        iou_max = 0
        for bbox_yolo in labels_yolo:
            bboxB = bbox_yolo[1:].copy()
            b_xc, b_yc, b_w, b_h = bboxB
            b_xc, b_w = b_xc * width, b_w * width
            b_yc, b_h = b_yc * height, b_h * height
            bboxB = self.xcycwh_to_xyxy([b_xc, b_yc, b_w, b_h])
            iou = self.bb_intersection_over_union(bboxA, bboxB)
            if iou > iou_max:
                iou_max = iou
        return iou_max

    def random_bbox(self, img, obj_width, obj_height):
        """
        random coordinates for bbox object in image
        :param img: original image
        :param obj_width: width of bbox object
        :param obj_height: height of bbox object
        """
        height, width = img.shape[:2]
        x_tl = np.random.randint(low=1, high=width - obj_width - 1)
        y_tl = np.random.randint(low=1, high=height - obj_height - 1)
        x_br = x_tl + obj_width
        y_br = y_tl + obj_height
        return [x_tl, y_tl, x_br, y_br]

    def get_random_perspective_transform(self,
                                         img_shape: tuple,
                                         degrees=20,
                                         translate=0.4,
                                         scale=0.4,
                                         shear=10,
                                         perspective=0.0) -> np.ndarray:
        height = img_shape[0]
        width = img_shape[1]

        # Center
        C = np.eye(3)
        C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        return M

    def warp_bbox(self, bbox: list, warp_mat: np.ndarray) -> list:
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1, x1, x2, x2],
                            [y1, y2, y1, y2],
                            [1,  1,  1,  1]])
        new_corners = warp_mat @ corners

        new_x1 = new_corners[0, :].min()
        new_x2 = new_corners[0, :].max()
        new_y1 = new_corners[1, :].min()
        new_y2 = new_corners[1, :].max()

        new_bbox = [new_x1, new_y1, new_x2, new_y2]
        return new_bbox

    def merge_images_weighted(self,
                              back_img: np.ndarray,
                              top_img: np.ndarray,
                              mask: np.ndarray,
                              weight: float = 1.0) -> np.ndarray:

        # Get weighted object image
        back_obj_img = cv2.bitwise_and(back_img, back_img, mask=mask)
        top_obj_img = cv2.bitwise_and(top_img, top_img, mask=mask)
        sum_obj_img = cv2.addWeighted(back_obj_img, 1 - weight, top_obj_img, weight, 0)

        # place weighted object image on back_img
        mask_inv = cv2.bitwise_not(mask)
        back_img_without_obj = cv2.bitwise_and(back_img, back_img, mask=mask_inv)
        final_img = cv2.add(back_img_without_obj, sum_obj_img)

        return final_img

    def xyxy_to_xywh(self, bbox):
        x_tl, y_tl, x_br, y_br = bbox
        return x_tl, y_tl, x_br - x_tl, y_br - y_tl

    def xyxy_to_xcycwh(self, bbox):
        x_tl, y_tl, x_br, y_br = bbox
        w, h = x_br - x_tl, y_br - y_tl
        return x_tl + w//2, y_tl + h//2, w, h

    def xywh_to_xcycwh(self, bbox):
        x_tl, y_tl, w, h = bbox
        return x_tl + w//2, y_tl + h//2, w, h
        
    def xcycwh_to_xyxy(self, bbox):
        x_c, y_c, w, h = bbox
        return x_c - w//2, y_c - h//2, x_c + w//2, y_c + h//2
        
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def read_labels(path: str) -> np.ndarray:
    with open(path) as f:
        rows = f.read().split('\n')

    lines = np.zeros((0, 5))
    for row in rows:
        if row == '':
            continue
        line = list(map(float, row.split(' ')))
        np.append(lines, np.array([line]), axis=0)

    return np.array(lines)


def write_labels(path: str, lines: np.ndarray):
    with open(path, 'w') as f:
        for i in range(lines.shape[0]):
            line = list(lines[i])
            f.write(' '.join(list(map(str, line))) + '\n')


def get_tube_crop(detector: Yolov5Detector, img: np.ndarray) -> np.ndarray:
    output = detector(img[:, :, ::-1])

    for i in range(output.shape[0]):
        x, y, w, h, cls_conf, cls_ids = output[i]

        if cls_conf > 0.4:
            print(w, h)
            tube_crop = img[int(y):int(y + h), int(x):int(x + w)]

            return tube_crop
    
    return None


if __name__ == '__main__':
    cnt = 0

    dataset_dir = r'F:\datasets\tmk_yolov5_25092022'
    new_dataset_dir = r'F:\datasets\tmk_yolov5_25092022_aug'
    
    detector = Yolov5Detector(r'E:\PythonProjects\AnnotationConverter\weights\last_tube.pt')
    golf = GOLF_Albumentations()
    
    splits = ['train']
    for split in splits:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')

        new_images_dir = os.path.join(new_dataset_dir, split, 'images')
        new_labels_dir = os.path.join(new_dataset_dir, split, 'labels')

        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_labels_dir, exist_ok=True)

        img_files = os.listdir(images_dir)
        img_files.sort()
        for img_file in img_files:
            img_name = os.path.splitext(img_file)[0]
            print(cnt, split, img_name)
            img = cv2.imread(os.path.join(images_dir, img_file))
            labels = read_labels(os.path.join(labels_dir, img_name + '.txt'))

            tube_img = get_tube_crop(detector, img)
            if tube_img is None:
                continue

            new_img, new_labels = golf(tube_img, labels, num_obj=8)

            cv2.imwrite(os.path.join(new_images_dir, img_name + '_aug.jpg'), new_img)
            write_labels(os.path.join(new_labels_dir, img_name + '_aug.txt'), new_labels)

            cnt += 1

            # break ### TMP

