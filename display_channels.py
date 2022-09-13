import os
import numpy as np
import cv2


img = cv2.imread(r'F:\datasets\tmk_09_09_2022\train\images\1_comet_2.jpg')
channels = cv2.split(img)
for i in range(len(channels)):
    cv2.imshow(str(i), cv2.resize(channels[i], (400, 400)))
cv2.waitKey()


