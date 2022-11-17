import cv2
import os
import glob


image_dir = ''
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (600, 600))
    phi, rho, gray = cv2.split(img)
    cv2.imshow('phi', phi)
    cv2.imshow('rho', rho)
    cv2.imshow('gray', gray)
    cv2.waitKey()



