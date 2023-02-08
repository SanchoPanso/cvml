import os
import sys
import math
import glob
import logging
import logging.config
import random
import cv2
import numpy as np
from typing import List
from pathlib import Path
import cProfile

sys.path.append(str(Path(__file__).parent.parent))

import cvml
from cvml.detection.augmentation.mask_cut_mix import MaskCutMix
from cvml.dataset.image_transforming import convert_to_mixed


def get_tube_img(img: np.ndarray) -> np.ndarray:
    
    gray = cv2.split(img)[2] #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 4)
    ret, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_TRIANGLE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    max_area = 0
    max_id = -1
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max_id = i
    
    rect = cv2.minAreaRect(contours[max_id])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    crop_box = np.array(
        [[0, 0],
         [0, 535],
         [640, 535],
         [640, 0]],
        dtype=np.int64        
    )
    warp_mat = cv2.getPerspectiveTransform(box.astype(np.float32), crop_box.astype(np.float32))
    crop_img = cv2.warpPerspective(img, warp_mat, img.shape[1::-1])
    
    return crop_img


if __name__ == '__main__':
    
    logging.config.dictConfig(cvml.LOGGER_CONFIG)
    logger = logging.getLogger('cvml')
    
    augmenter = MaskCutMix(r'C:\Users\HP\Downloads\defects_segment_27092022', 
                           ['comet'], 
                           ['comet'])
    
    cvs1_comet_dir = r'C:\Users\HP\Downloads\csv1_comet_1\csv1_comet_1\images'
    img_paths = set(glob.glob(os.path.join(cvs1_comet_dir, '*'))) - set(glob.glob(os.path.join(cvs1_comet_dir, '*color*')))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = convert_to_mixed(img)
        img = cv2.resize(img, (640, 535))
        
        img = get_tube_img(img)
        labels = np.array([[0.0, 0.1, 0.1, 0.05, 0.05]])
        
        # new_img, new_labels = augmenter(img, labels)
        
        # for label in new_labels:
        #     cls_id, xc, yc, w, h = label
        #     xc *= img.shape[1]
        #     yc *= img.shape[0]
        #     w *= img.shape[1]
        #     h *= img.shape[0]
            
        #     cv2.rectangle(new_img, 
        #                   (int(xc - w/2), int(yc - h/2)), 
        #                   (int(xc + w/2), int(yc + h/2)), 
        #                   (0, 255, 0), 1)
        
        # cv2.imshow('new_img', new_img)
        # cv2.waitKey()
        cProfile.run("augmenter(img, labels)",  sort="time")        
        break