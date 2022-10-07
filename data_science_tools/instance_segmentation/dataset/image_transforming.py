import os
import cv2
import numpy as np
from typing import List
from array import array


def get_mask_polygon(mask: np.ndarray) -> List[List[List[float]]]:
    """
    Convert color mask into the set of points, which is the set of corners of apporximating polygon
    :mask: bgr image, in which black pixels - the absence of objects, other - the appearance of object
    """

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    edged_mask = cv2.Canny(binary_mask, 50, 150)
    contours, hierarchy = cv2.findContours(edged_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for point in contours[0].tolist():
        x, y = point[0]
        points.append(x)
        points.append(y)

    return points


if __name__ == '__main__':
    img = np.zeros((400, 400, 3), dtype='uint8')
    img = cv2.rectangle(img, (80, 80), (320, 320), (0, 0, 200), -1)
    img = cv2.rectangle(img, (10, 10), (100, 100), (0, 0, 200), -1)
    points = get_mask_polygon(img)

    print(points)
    for i in range(0, len(points), 2):
        x1, y1 = points[i], points[i + 1]
        x2, y2 = points[(i + 2) % len(points)], points[(i + 2) % len(points)]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)            
    
    cv2.imshow("test", img)
    cv2.waitKey()

