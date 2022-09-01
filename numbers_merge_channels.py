import os
import cv2
import numpy as np
import glob
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


datasets_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets')
numbers_1_result = os.path.join(datasets_dir, 'numbers1_merged')
numbers_2_result = os.path.join(datasets_dir, 'numbers2_merged')


numbers_1_dirs = [
    r'C:\Users\Alex\Downloads\number1_polar\number1_segment',
    r'C:\Users\Alex\Downloads\number1_polar\number1_segment',
    r'C:\Users\Alex\Downloads\Nomera_defecti\Nomera_defecti\numbers1\numbers1',
]

postfix_1 = [
    '_1.png',
    '_2.png',
    '.png',
]

numbers_2_dirs = [
    r'C:\Users\Alex\Downloads\23_06_2021_polar\23_06_2021_polar',
    r'C:\Users\Alex\Downloads\23_06_2021_polar\23_06_2021_polar',
    r'C:\Users\Alex\Downloads\23_06_2021_polar\23_06_2021_polar\gray',
]

postfix_2 = [
    '_1.png',
    '_2.png',
    '.png',
]

numbers_1_files_3 = glob.glob(os.path.join(numbers_1_dirs[2], '*.png'))
numbers_2_files_3 = glob.glob(os.path.join(numbers_2_dirs[2], '*.png'))


numbers_1_names = [os.path.splitext(os.path.split(i)[-1])[0] for i in numbers_1_files_3]
numbers_2_names = [os.path.splitext(os.path.split(i)[-1])[0] for i in numbers_2_files_3]


# os.makedirs(numbers_1_result, exist_ok=True)
# for name in numbers_1_names:
#     imgs = [None for i in range(3)]
#
#     for i in range(3):
#         print(name + postfix_1[i])
#         imgs[i] = cv2.imread(os.path.join(numbers_1_dirs[i], name + postfix_1[i]))
#         if i == 2:
#             imgs[i] = expo(imgs[i], 20)
#         imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
#
#     final_img = cv2.merge(imgs)
#     cv2.imwrite(os.path.join(numbers_1_result, name + '.jpg'), final_img)


os.makedirs(numbers_2_result, exist_ok=True)
for name in numbers_2_names:
    imgs = [None for i in range(3)]

    for i in range(3):
        print(name + postfix_1[i])
        imgs[i] = cv2.imread(os.path.join(numbers_2_dirs[i], name + postfix_2[i]))
        if i == 2:
            imgs[i] = expo(imgs[i], 20)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)

    final_img = cv2.merge(imgs)
    cv2.imwrite(os.path.join(numbers_2_result, name + '.jpg'), final_img)


