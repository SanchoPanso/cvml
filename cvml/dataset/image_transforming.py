import os
import numpy as np
import cv2
import math
import torch
from typing import List
from cvml.augmentation.sp_estimator import SPEstimator, SPEstimatorNumpy


def expo(img: np.ndarray, step: int) -> np.ndarray:
    
    lut = np.zeros((256,), dtype='uint8')
    for i in range(256):
        lut_i = i + math.sin(i * 0.01255) * step * 10
        lut[i] = int(max(0, min(255, lut_i)))

    if len(img.shape) == 2: # Grayscale case
        img = cv2.LUT(img, lut)
        
    else:   # RGB case
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        hsv = cv2.split(img)
        hsv = (hsv[0], hsv[1], cv2.LUT(hsv[2], lut))   
             
        img = cv2.merge(hsv)                                 
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)     
                
    return img


def quadratic_lightening(img: np.ndarray, coef: float = 2.35):
    lut = np.zeros((256,), dtype='uint8')
    for i in range(256):
        lut_i = i + coef * (i - i**2 / 255)
        lut[i] = np.clip(lut_i, 0, 255)

    if len(img.shape) == 2: # Grayscale case
        img = cv2.LUT(img, lut)
        
    else:   # RGB case
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        hsv = cv2.split(img)
        hsv = (hsv[0], hsv[1], cv2.LUT(hsv[2], lut))   
             
        img = cv2.merge(hsv)                                 
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)     
                
    return img



def normalize_min_max(data):
    data_min = data.min()
    data_max = data.max()
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data


def convert_to_mixed(orig_img: np.ndarray, estimator: SPEstimatorNumpy = None) -> np.ndarray:

    height, width = orig_img.shape[0:2]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #in_data = torch.from_numpy(img).float() #torch.frombuffer(orig_img.data, dtype=torch.uint8, count=img.size).float().detach_().reshape(height, width)
    
    estimator = estimator or SPEstimatorNumpy()
    rho, phi = estimator.getAzimuthAndPolarization(img)
    
    normalized_rho = normalize_min_max(rho)
    normalized_phi = normalize_min_max(phi)

    rho_img = (normalized_rho * 255).astype('uint8')
    phi_img = (normalized_phi * 255).astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray_img = expo(img, 15)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    img = cv2.merge([phi_img, rho_img, gray_img])
    return img


def get_lines(img: np.ndarray):
    dst = cv2.medianBlur(img, 21)
    dst = cv2.Canny(dst, 10, 50, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    return lines


def get_mask_contours(mask: np.ndarray) -> tuple:
    """
    Convert color mask into the set of points, which is the set of corners of apporximating polygon
    :mask: bgr image, in which black pixels - the absence of objects, other - the appearance of object
    """

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


if __name__ == '__main__':
    
    img = cv2.imread(r'D:\datasets\tmk_yolov5_25092022\train\images\1_comet_3.jpg')
    img = cv2.split(img)[2]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lines = get_lines(img)
    print(lines)
    print(len(lines))

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
            pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("test", cv2.resize(img, (400, 400)))
    cv2.waitKey()
    
    
    img = np.zeros((400, 400, 3), dtype='uint8')
    img = cv2.rectangle(img, (80, 80), (180, 180), (0, 0, 200), -1)
    img = cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 200), -1)
    
    cv2.circle(img, (200, 200), 10, (0, 200, 0), -1)
    
    cv2.polylines(img, (np.array([[[300, 300], [320, 300], [330, 310], [320, 320], [310, 320]]], dtype=np.int32)), True, (200, 0, 0), 1)
    
    contours = get_mask_contours(img)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)           
    
    cv2.imshow("test", img)
    cv2.waitKey()

