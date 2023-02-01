import os
import sys
import cv2
import glob
import numpy as np


def get_tube_img(img: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
         [0, 640],
         [640, 640],
         [640, 0]],
        dtype=np.int64        
    )
    warp_mat = cv2.getPerspectiveTransform(box.astype(np.float32), crop_box.astype(np.float32))
    crop_img = cv2.warpPerspective(img, warp_mat, img.shape[1::-1])
    
    return crop_img


cvs1_comet_dir = r'C:\Users\HP\Downloads\csv1_comet_1\csv1_comet_1\images'
img_paths = set(glob.glob(os.path.join(cvs1_comet_dir, '*'))) - set(glob.glob(os.path.join(cvs1_comet_dir, '*color*')))
for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 535))
    
    crop_img = get_tube_img(img)
    
    cv2.imshow(os.path.split(img_path)[-1], img)
    cv2.imshow("crop", crop_img)
    if cv2.waitKey() == 27:
        break
    cv2.destroyAllWindows()