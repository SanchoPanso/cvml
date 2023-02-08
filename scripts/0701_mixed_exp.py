import os
import sys
import glob
import logging
import cv2
import numpy as np

sys.path.append(os.path.dirname(__file__) + '/..')
from cvml.dataset.image_transforming import convert_to_mixed


img = cv2.imread(r'D:\CodeProjects\PythonProjects\TMK_CSV1_detection\TMK_CSV1_detection\images\223.png')
img = convert_to_mixed(img)
phi, rho, gray = cv2.split(img)
new_phi = phi.copy()

for i in range(new_phi.shape[0]):
    for j in range(new_phi.shape[1]):
        if rho[i][j] < 32:
            new_phi[i][j] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# new_phi = cv2.morphologyEx(new_phi, cv2.MORPH_CLOSE, kernel)
# new_phi = cv2.dilate(new_phi, kernel)

cv2.imshow("rho", cv2.resize(rho, (600, 600)))
cv2.imshow("phi", cv2.resize(phi, (600, 600)))
cv2.imshow("new_phi", cv2.resize(new_phi, (600, 600)))
cv2.imshow("gray", cv2.resize(gray, (600, 600)))

cv2.waitKey()