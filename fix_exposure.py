import os

import cv2
import numpy as np
import math


def expo(img: np.ndarray, step: int):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    lut = GetGammaExpo(step)
    hsv = cv2.split(img)
    hsv = (hsv[0], hsv[1], cv2.LUT(hsv[2], lut))
    img = cv2.merge(hsv)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB_FULL)

    return img


def GetGammaExpo(step: int):

    result = np.zeros((256), dtype='uint8');

    for i in range(256):
        result[i] = AddDoubleToByte(i, math.sin(i * 0.01255) * step * 10)

    return result


def AddDoubleToByte(bt: int, d: float):
    result = bt
    if float(result) + d > 255:
        result = 255
    elif float(result) + d < 0:
        result = 0
    else:
        result += d
    return result


if __name__ == '__main__':
    orig_image_dir = r'C:\Users\Alex\Downloads\Nomera_defecti\Nomera_defecti\numbers1\numbers1'
    result_image_dir = r'E:\PythonProjects\AnnotationConverter\datasets\numbers_raw'

    os.makedirs(result_image_dir, exist_ok=True)

    files = os.listdir(orig_image_dir)
    files.sort()
    for file in files:
        print(file)
        img = cv2.imread(os.path.join(orig_image_dir, file))
        exp_img = expo(img, 20)
        cv2.imwrite(os.path.join(result_image_dir, file), exp_img)

