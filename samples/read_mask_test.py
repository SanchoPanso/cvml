import cv2
import numpy as np


path = r'D:\datasets_old\segment_25082022\train\4_2_2.npz'

with np.load(path) as data:
    arr = data['arr_0']
print(arr)
print(arr.shape)
