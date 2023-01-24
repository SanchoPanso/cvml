import os
import cv2
import numpy as np
from typing import List
from array import array


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
    img = np.zeros((400, 400, 3), dtype='uint8')
    img = cv2.rectangle(img, (80, 80), (180, 180), (0, 0, 200), -1)
    img = cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 200), -1)
    
    cv2.circle(img, (200, 200), 10, (0, 200, 0), -1)
    
    cv2.polylines(img, (np.array([[[300, 300], [320, 300], [330, 310], [320, 320], [310, 320]]], dtype=np.int32)), True, (200, 0, 0), 1)
    
    contours = get_mask_contours(img)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)           
    
    cv2.imshow("test", img)
    cv2.waitKey()

