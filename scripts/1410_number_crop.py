import os
import glob
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.annotation.bounding_box import BoundingBox
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.dataset.image_transforming import expo
from cvml.detection.dataset.io_handling import read_yolo_labels



if __name__ == '__main__':
    dataset = r'D:\datasets\tmk_yolov5_25092022'
    new_dataset = r'D:\datasets\tmk_cvs1_14102022_num_crops'
    for split in ['train', 'valid']:
        img_dir = os.path.join(dataset, split, 'images')
        txt_dir = os.path.join(dataset, split, 'labels')

        img_files = os.listdir(img_dir)
        txt_files = os.listdir(txt_dir)

        mixed_res_dir = os.path.join(new_dataset, split, 'mixed') 
        gray_res_dir = os.path.join(new_dataset, split, 'gray')
        phi_res_dir = os.path.join(new_dataset, split, 'phi')
        rho_res_dir = os.path.join(new_dataset, split, 'rho')

        os.makedirs(mixed_res_dir, exist_ok=True)
        os.makedirs(gray_res_dir, exist_ok=True)
        os.makedirs(phi_res_dir, exist_ok=True)
        os.makedirs(rho_res_dir, exist_ok=True)

        for img_file in img_files:
            name, ext = os.path.splitext(img_file)
            txt_file = name + '.txt'
            bboxes = read_yolo_labels(os.path.join(txt_dir, txt_file), (2448, 2048))
            img = cv2.imread(os.path.join(img_dir, img_file))

            crop_cnt = 1
            for bbox in bboxes:
                if bbox.get_class_id() != 3:
                    continue
                x, y, w, h = bbox.get_absolute_bounding_box()
                
                img_crop = img[y: y + h, x: x + w]
                phi_crop, rho_crop, gray_crop = cv2.split(img_crop)
                
                crop_name = f'{name}_{crop_cnt}.jpg'
                cv2.imwrite(os.path.join(mixed_res_dir, crop_name), img_crop)
                cv2.imwrite(os.path.join(gray_res_dir, crop_name), cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR))
                cv2.imwrite(os.path.join(phi_res_dir, crop_name), cv2.cvtColor(phi_crop, cv2.COLOR_GRAY2BGR))
                cv2.imwrite(os.path.join(rho_res_dir, crop_name), cv2.cvtColor(rho_crop, cv2.COLOR_GRAY2BGR))

                crop_cnt += 1
                print(img_file)
                # cv2.imshow("test", cv2.resize(cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5), (400, 400)))
                # cv2.waitKey()





