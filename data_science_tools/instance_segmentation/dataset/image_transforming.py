import os
import cv2
import numpy as np
from typing import List
from array import array


def get_mask_polygon(mask: np.ndarray):
    """
    Convert color mask into the set of points, which is the set of corners of apporximating polygon
    :mask: bgr image, in which black pixels - the absence of objects, other - the appearance of object
    """

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


if __name__ == '__main__':
    img = np.zeros((400, 400, 3), dtype='uint8')
    img = cv2.rectangle(img, (80, 80), (320, 320), (255, 255, 255), -1)
    contours = get_mask_polygon(img)
    print(contours)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow("test", img)
    cv2.waitKey()

