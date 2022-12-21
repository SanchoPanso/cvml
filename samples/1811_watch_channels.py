import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


image_dir = '/home/student2/datasets/prepared/tmk_cvs1_yolov5_31102022/train/images'
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (600, 600))

    phi, rho, gray = cv2.split(img)
    k = 0.7
    lut = [max(0, min(255, int(k*i + 256 * k / 2))) for i in range(256)]
    phi = cv2.LUT(phi, np.array(lut, dtype='uint8'))

    cv2.imshow('phi', phi)
    cv2.imshow('rho', rho)
    cv2.imshow('gray', gray)

    #hist = cv2.calcHist(phi, [0], None, [32], [0, 256])
    #break

    if cv2.waitKey() == 27:
        break



