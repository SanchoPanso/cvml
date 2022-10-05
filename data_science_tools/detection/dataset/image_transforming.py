import numpy as np
import cv2
import math


def expo(img: np.ndarray, step: int) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)     # convert rgb to hsv
    lut = get_gamma_expo(step)                          # get look-up table (array) of 256 elems
    hsv = cv2.split(img)                                # get tuple of 3 channels of hsv-format
    hsv = (hsv[0], hsv[1], cv2.LUT(hsv[2], lut))        # apply look-up table transform for the value channel
    img = cv2.merge(hsv)                                # merge channels to img 
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB_FULL)     # convert back to rgb format

    return img


def get_gamma_expo(step: int) -> np.ndarray:
    """get look-up table"""

    result = np.zeros((256,), dtype='uint8')

    for i in range(256):
        result[i] = add_double_to_byte(i, math.sin(i * 0.01255) * step * 10)

    return result


def add_double_to_byte(bt: int, d: float) -> int:
    """summarize in range of (0, 255)"""
    result = bt
    if float(result) + d > 255:
        result = 255
    elif float(result) + d < 0:
        result = 0
    else:
        result += d
    return result


