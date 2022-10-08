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


def get_lines(img: np.ndarray):
    dst = cv2.medianBlur(img, 21)
    dst = cv2.Canny(dst, 10, 50, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    return lines


if __name__ == '__main__':
    img = cv2.imread(r'D:\datasets\tmk_yolov5_25092022\train\images\1_comet_3.jpg')
    img = cv2.split(img)[2]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lines = get_lines(img)
    print(lines)
    print(len(lines))

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
            pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("test", cv2.resize(img, (400, 400)))
    cv2.waitKey()


