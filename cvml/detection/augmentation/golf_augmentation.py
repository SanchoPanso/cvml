import os
import sys
import logging
import math
import random
import time
from typing import Tuple, List

import cv2
import numpy as np


class MaskMixup:
    def __init__(self, 
                 crop_obj_dir: str, 
                 crop_class_names: List[str], 
                 class_names: List[str]):

        self.dir_coco_obj = crop_obj_dir
        self.coco_class_names = crop_class_names
        self.class_names = class_names

        self.coco_conformity = {cls_name: i for i, cls_name in enumerate(class_names)}  # CHECK

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

    def __call__(self, im, labels, p=1.0, num_obj=7, **kwargs):
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
        distortion_warp_mat = self.get_random_perspective_transform(img.shape, translate=0.0)
        new_bbox = None
        for sample_id in range(10):
            translate_warp_mat = self.get_random_perspective_transform(img.shape, 
                                                                       degrees=0,
                                                                       translate=0.4, 
                                                                       scale=0.0, 
                                                                       shear=0, 
                                                                       perspective=0.0)
            warp_mat = translate_warp_mat @ distortion_warp_mat
            new_bbox = self.warp_bbox(bbox, warp_mat)
            iou_max = self.get_iou_max(img, new_bbox, labels_yolo)
            if iou_max <= 0.1:
                break
        if iou_max > 0.1:
            return img, None
        
        # start = time.time()
        
        x1, y1, x2, y2 = map(int, new_bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1] - 1, x2)
        y2 = min(img.shape[0] - 1, y2)

        # # place img_obj on black background with size of original img
        # img_obj_on_black = np.zeros(img.shape, dtype=img.dtype)
        # img_obj_on_black[y_tl:y_br, x_tl:x_br] = img_obj
        # new_img_obj = cv2.warpPerspective(img_obj_on_black, warp_mat, (width, height))

        # opacity = random.uniform(0.3, 0.7)
        # final_img = cv2.addWeighted(img, 1, new_img_obj, opacity, 0)
        
        # end = time.time()
        # print(end - start)

        # place img_obj on black background with size of original img
        img_obj_on_black = np.zeros(img.shape, dtype=img.dtype)
        img_obj_on_black[y_tl:y_br, x_tl:x_br] = img_obj

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
        
        # end = time.time()
        # print(end - start)
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
            img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)  # CHECK
            return img_obj
        except Exception as e:
            return None

    def select_obj(self):
        # ENDLESS LOOP!!!
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
                                         scale=0.45,
                                         shear=30,
                                         perspective=0.0,
                                         scale_coef=0.6) -> np.ndarray:
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
        s = random.uniform(1 - scale, 1 + scale) * scale_coef # special rescale
        # s = random.uniform(1 - scale, 1 + scale)
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

    @staticmethod
    def warp_bbox(bbox: list, warp_mat: np.ndarray) -> list:
        x1, y1, x2, y2 = bbox
        corners = np.single([[[x1, y1],
                              [x1, y2],
                              [x2, y1],
                              [x2, y2]]])
        new_corners = cv2.perspectiveTransform(corners, warp_mat)

        new_x1 = new_corners[0, :, 0].min()
        new_x2 = new_corners[0, :, 0].max()
        new_y1 = new_corners[0, :, 1].min()
        new_y2 = new_corners[0, :, 1].max()

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

    lines = []
    for row in rows:
        if row == '':
            continue
        line = list(map(float, row.split(' ')))
        lines.append(line)

    return np.array(lines).reshape(-1, 5)


def write_labels(path: str, lines: np.ndarray):
    with open(path, 'w') as f:
        for i in range(lines.shape[0]):
            line = list(lines[i])
            f.write(' '.join(list(map(str, line))) + '\n')


# def get_tube_crop(detector: Yolov5Detector, img: np.ndarray) -> np.ndarray:
#     output = detector(img[:, :, ::-1])
#     if output.shape[0] == 0:
#         return None
    
#     x, y, w, h, _, _ = output[0]
#     tube_crop = img[y: y + h, x: x + w]

#     return tube_crop


def test_line(img):
    src = img.copy()
    src = cv2.split(src)[2]
    
    src = cv2.GaussianBlur(src, (5, 5), 0)
    dst = cv2.Canny(src, 50, 100, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 60, None, 0, 0, np.pi*0.55, np.pi*0.7)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv2.waitKey(0)


def constrain(value, lower_bound, upper_bound):
    if value < lower_bound:
        return lower_bound
    elif value > upper_bound:
        return upper_bound
    return value


def get_intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def find_line_bounds(line, height, width):
    h_border_1 = [[0, 0.5 * np.pi]]
    h_border_2 = [[height, 0.5 * np.pi]]
    
    v_border_1 = [[0, 0]]
    v_border_2 = [[width, 0]]

    rho, theta = line[0]
    
    points = []
    
    if abs(theta) < 0.001:
        intersection_1 = get_intersection(line, h_border_1)
        intersection_2 = get_intersection(line, h_border_2)

        intersection_1[0] = constrain(intersection_1[0], 0, width)
        intersection_2[0] = constrain(intersection_2[0], 0, width)

        points = [intersection_1, intersection_2]
    
    elif abs(theta - 0.5 * np.pi) < 0.001:
        intersection_1 = get_intersection(line, v_border_1)
        intersection_2 = get_intersection(line, v_border_2)

        intersection_1[1] = constrain(intersection_1[1], 0, height)
        intersection_2[1] = constrain(intersection_2[1], 0, height)

        points = [intersection_1, intersection_2]
    else:
        intersections = [0, 0, 0, 0]
        intersections[0] = get_intersection(line, v_border_1)
        intersections[1] = get_intersection(line, v_border_2)
        intersections[2] = get_intersection(line, h_border_1)
        intersections[3] = get_intersection(line, h_border_2)

        for i in range(4):
            x, y = intersections[i]
            if 0 <= x <= width and 0 <= y <= height:
                points.append([x, y])
            
            if len(points) == 2:
                break

    return points


def find_lines_by_params(dst, min_theta, max_theta, min_rho_delta, max_rho_delta):
    # 0.06 * np.pi, 0.18 * np.pi
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 63, None, 0, 0, min_theta, max_theta)    # -0.02 * np.pi, 0.02 * np.pi
    lines = [] if lines is None else [[[lines[i][0][0], lines[i][0][1]]] for i in range(len(lines))]
    lines.sort(key=lambda x: x[0][0])

    target_lines = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1 = lines[i][0][0]
            theta1 = lines[i][0][1]
            
            rho2 = lines[j][0][0]
            theta2 = lines[j][0][1]

            if min_rho_delta <= rho2 - rho1 <= max_rho_delta:
                print(rho1, rho2)
                target_lines = [lines[i], lines[j]] 
                break

    return target_lines


def get_tube_crop(img: np.ndarray, labels: np.ndarray, height: int, width: int) -> np.ndarray:
    src = img.copy()
    gray = cv2.split(src)[2]
    
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    dst = cv2.Canny(gray, 50, 100, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    params_list = [
        {'min_theta': 0.06 * np.pi, 'max_theta': 0.18 * np.pi, 'min_rho_delta': 300, 'max_rho_delta': 500},
        {'min_theta': -0.02 * np.pi, 'max_theta': 0.02 * np.pi, 'min_rho_delta': 300, 'max_rho_delta': 500},
        {'min_theta': 0.55 * np.pi, 'max_theta': 0.7 * np.pi, 'min_rho_delta': 300, 'max_rho_delta': 500},
    ]

    for params in params_list:
        target_lines = find_lines_by_params(dst, **params)
        if len(target_lines) != 0:
            break
    
    if len(target_lines) == 0:
        return False, img, labels
    

    region_points = []
    if target_lines is not None:
        for i in range(0, len(target_lines)):
            rho = target_lines[i][0][0]
            theta = target_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

            ipt1, ipt2 = find_line_bounds(target_lines[i], cdst.shape[0], cdst.shape[1])
            region_points.append(ipt1)
            region_points.append(ipt2)
            cv2.circle(cdst, (ipt1[0], ipt1[1]), 5, (0, 255, 0), -1)
            cv2.circle(cdst, (ipt2[0], ipt2[1]), 5, (0, 255, 0), -1)
    
    if len(region_points) != 0:
        src_region_points = np.array(region_points, dtype=np.float32)
        dst_region_points = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        
        warp_mat = cv2.getPerspectiveTransform(src_region_points, 
                                               dst_region_points)
        tube_img = cv2.warpPerspective(src, warp_mat, (width, height))
        
        new_bboxes = []
        for bbox in labels:
            cls_id, x, y, w, h = bbox
            bbox_xyxy = [(x - w/2) * width, (y - h/2) * height, (x + w/2) * width, (y + h/2) * height]
            new_xyxy = MaskMixup.warp_bbox(bbox_xyxy, warp_mat)

            x1, y1, x2, y2 = new_xyxy

            x1 /= width
            y1 /= height
            x2 /= width
            y2 /= height

            if x2 < 0 or y2 < 0:
                continue
            if x1 > 1 or  y1 > 1:
                continue
            
            x1 = max(0, x1)
            x2 = min(1, x2)
            y1 = max(0, y1)
            y2 = min(1, y2)

            w = x2 - x1
            h = y2 - y1

            new_bboxes.append([cls_id, x1 + w/2, y1 + h/2, w, h])
        
        return True, tube_img, np.array(new_bboxes).reshape(-1, 5)

    return False, img, labels




class MaskMixupAugmentation:

    def __call__(self, 
                 mask_mixup_augmenter: MaskMixup, 
                 img: np.ndarray, 
                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray] or None:
        pass


class MandrelAugmentation(MaskMixupAugmentation):

    def __call__(self, 
                 mask_mixup_augmenter: MaskMixup, 
                 img: np.ndarray, 
                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray] or None:
        
        height, width = img.shape[0], img.shape[1]
        ret, tube_img, tube_labels = get_tube_crop(img, labels, height, width)
        if ret is False:
            return None
        
        new_tube_img, new_tube_labels = mask_mixup_augmenter(tube_img, tube_labels)
        return new_tube_img, new_tube_labels


class TubeAugmentation(MaskMixupAugmentation):

    def __call__(self, 
                 mask_mixup_augmenter: MaskMixup, 
                 img: np.ndarray, 
                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray] or None:
        
        new_tube_img, new_tube_labels = mask_mixup_augmenter(img, labels)
        return new_tube_img, new_tube_labels



if __name__ == '__main__':

    dataset_dir = '/home/student2/datasets/prepared/tmk_cvs3_yolov5_17122022'
    new_dataset_dir = '/home/student2/datasets/prepared/tmk_cvs3_yolov5_17122022_aug'
    
    # detector = Yolov5Detector(r'E:\PythonProjects\AnnotationConverter\weights\yolov5l_tube.pt')
    dir_coco_obj = "/home/student2/datasets/crops/0210_defect_crops"#"/home/student2/datasets/crops/0712_comet_crops"
    coco_class_names = ['comet', 'other', 'joint', 'number', 'tube', 'sink', 'birdhouse', 'print', 'riska', 'deformation defect', 'continuity violation']#['comet']
    class_names = ['comet', 'other', 'joint', 'number', 'tube', 'sink', 'birdhouse', 'print', 'riska', 'deformation defect', 'continuity violation']#['comet']
    golf = MaskMixup(crop_obj_dir=dir_coco_obj, crop_class_names=coco_class_names, class_names=class_names)
    
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
            print(split, img_name)
            img = cv2.imread(os.path.join(images_dir, img_file))
            img = cv2.resize(img, (640, 535))
            labels = read_labels(os.path.join(labels_dir, img_name + '.txt'))
            # print('old_labels', labels)

            if 3 in [labels[i][0] for i in range(labels.shape[0])]:
                print('3 in', img_name)
                continue
            
            labels = np.zeros((0, 5))
            # ret, img, labels = get_tube_crop(img, labels, img.shape[0], img.shape[1])
            # if ret is False:
            #     continue
            
            # tube_img = get_tube_crop(detector, img)
            new_img, new_labels = golf(img, labels)

            # print(new_labels)
            
            # for i in range(new_labels.shape[0]):
            #     cls_id, xc, yc, w, h = new_labels[i]
            #     x = int((xc - w/2) * img.shape[1])
            #     y = int((yc - h/2) * img.shape[0])
            #     w = int(w * img.shape[1])
            #     h = int(h * img.shape[0])
            #     cv2.rectangle(new_img,
            #                 (x, y),
            #                 (x + w, y + h),
            #                 (0, 255, 0), 5)
        
            # cv2.imshow("test", new_img)
            # if cv2.waitKey(0) == 27:
            #     break

            cv2.imwrite(os.path.join(new_images_dir, img_name + '_aug.jpg'), new_img)
            write_labels(os.path.join(new_labels_dir, img_name + '_aug.txt'), new_labels)

            # break ### TMP


