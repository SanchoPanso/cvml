import os
import cv2
import math
import shutil
import numpy as np
from typing import Callable
from detection.dataset_tools.coco_extractor import CocoExtractor
from detection.dataset_tools.label_editor import LabelEditor
from detection.dataset_tools.installer import Installer


tmk_dir = r'F:\TMK'

comet_1_images_dir = os.path.join(tmk_dir, 'csv1_comets_1_24_08_2022', 'images')
comet_1_polarization_dir = os.path.join(tmk_dir, 'csv1_comets_1_24_08_2022', 'polarization')

comet_2_images_dir = os.path.join(tmk_dir, 'csv1_comets_2_24_08_2022', 'images')
comet_2_polarization_dir = os.path.join(tmk_dir, 'csv1_comets_2_24_08_2022', 'polarization')

comet_3_images_dir = os.path.join(tmk_dir, 'csv1_comets_23_08_2022', 'images')
comet_3_polarization_dir = os.path.join(tmk_dir, 'csv1_comets_23_08_2022', 'polarization')

comet_4_images_dir = os.path.join(tmk_dir, 'csv1_comets_01_09_2022', 'images')
comet_4_polarization_dir = os.path.join(tmk_dir, 'csv1_comets_01_09_2022', 'polarization')

comet_5_images_dir = os.path.join(tmk_dir, 'csv1_comets_05_09_2022', 'images')
comet_5_polarization_dir = os.path.join(tmk_dir, 'csv1_comets_05_09_2022', 'polarization')

datasets_dir = r'F:\datasets'

comet_1_merged_dir = os.path.join(datasets_dir, 'comet_1_raw_merged')
comet_2_merged_dir = os.path.join(datasets_dir, 'comet_2_raw_merged')
comet_3_merged_dir = os.path.join(datasets_dir, 'comet_3_raw_merged')
comet_4_merged_dir = os.path.join(datasets_dir, 'comet_4_raw_merged')
comet_5_merged_dir = os.path.join(datasets_dir, 'comet_5_raw_merged')

number_0_images_dir = os.path.join(tmk_dir, 'numbers_23_06_2021', 'images')
number_0_polarization_dir = os.path.join(tmk_dir, 'numbers_23_06_2021', 'polarization')

number_1_images_dir = os.path.join(tmk_dir, 'numbers_24_08_2022', 'images')
number_1_polarization_dir = os.path.join(tmk_dir, 'numbers_24_08_2022', 'polarization')

number_2_images_dir = os.path.join(tmk_dir, 'numbers_25_08_2022', 'images')
number_2_polarization_dir = os.path.join(tmk_dir, 'numbers_25_08_2022', 'polarization')

number_3_images_dir = os.path.join(tmk_dir, 'numbers_01_09_2022', 'images')
number_3_polarization_dir = os.path.join(tmk_dir, 'numbers_01_09_2022', 'polarization')


number_0_merged_dir = os.path.join(datasets_dir, 'number_0_raw_merged')
number_1_merged_dir = os.path.join(datasets_dir, 'number_1_raw_merged')
number_2_merged_dir = os.path.join(datasets_dir, 'number_2_raw_merged')
number_3_merged_dir = os.path.join(datasets_dir, 'number_3_raw_merged')

comet_1_images_annotation_path = os.path.join(tmk_dir, 'csv1_comets_1_24_08_2022', 'annotations', 'instances_default.json')
comet_2_images_annotation_path = os.path.join(tmk_dir, 'csv1_comets_2_24_08_2022', 'annotations', 'instances_default.json')
comet_3_images_annotation_path = os.path.join(tmk_dir, 'csv1_comets_23_08_2022', 'annotations', 'instances_default.json')
comet_4_images_annotation_path = os.path.join(tmk_dir, 'csv1_comets_01_09_2022', 'annotations', 'instances_default.json')
comet_5_images_annotation_path = os.path.join(tmk_dir, 'csv1_comets_05_09_2022', 'annotations', 'instances_default.json')

number_0_annotation_path = os.path.join(tmk_dir, 'numbers_23_06_2021', 'annotations', 'instances_default_without_kiril.json')
number_1_annotation_path = os.path.join(tmk_dir, 'numbers_24_08_2022', 'annotations', 'instances_default.json')
number_2_annotation_path = os.path.join(tmk_dir, 'numbers_25_08_2022', 'annotations', 'instances_default.json')
number_3_annotation_path = os.path.join(tmk_dir, 'numbers_01_09_2022', 'annotations', 'instances_default.json')


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


def merge_images(img_paths: list) -> np.ndarray:
    imgs = []
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)

        if i == 2:
            img = expo(img, 15)  # orig must be exposure

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)

    merged_img = cv2.merge(imgs)
    return merged_img


def make_merged_dataset(images_dir: str, polarization_dir: str, install_dir: str):
    os.makedirs(install_dir, exist_ok=True)
    img_files = os.listdir(images_dir)

    for img_file in img_files:
        name, ext = os.path.splitext(img_file)
        polar_file_1 = name + '_1' + ext
        polar_file_2 = name + '_2' + ext

        img_path = os.path.join(images_dir, img_file)
        polar_1_path = os.path.join(polarization_dir, polar_file_1)
        polar_2_path = os.path.join(polarization_dir, polar_file_2)

        merged_img = merge_images([polar_2_path,  polar_1_path, img_path])
        cv2.imwrite(os.path.join(install_dir, name + '.jpg'), merged_img)
        print(install_dir, name)


def rename_dir_files(dir_path: str, rename_callback: Callable):
    files = os.listdir(dir_path)
    for file in files:
        name, ext = os.path.splitext(file)
        new_name = rename_callback(name)
        os.rename(os.path.join(dir_path, name + ext),
                  os.path.join(dir_path, new_name + ext))
        print(os.path.join(dir_path, name + ext),
              '- >',
              os.path.join(dir_path, new_name + ext))


def rename_dataset_files(dataset_dir: str, rename_callback: Callable):
    splits = ['train', 'valid', 'test']
    data_types = ['images', 'labels']
    for split in splits:
        for data_type in data_types:
            rename_dir_files(os.path.join(dataset_dir, split, data_type), rename_callback)


def rename_annotation_files(annotations_data: dict,  rename_callback: Callable, new_ext: str = None) -> dict:
    filenames = list(annotations_data['annotations'].keys())
    new_annotations_data = {'classes': annotations_data['classes'], 'annotations': {}}
    for filename in filenames:
        print(filename)
        name, ext = os.path.splitext(filename)
        new_name = rename_callback(name)

        if new_ext is None:
            new_filename = new_name + ext
        else:
            new_filename = new_name + new_ext

        labels = annotations_data['annotations'][filename]
        new_annotations_data['annotations'][new_filename] = labels

    return new_annotations_data


def make_merged_comets():
    make_merged_dataset(comet_1_images_dir, comet_1_polarization_dir, comet_1_merged_dir)
    make_merged_dataset(comet_2_images_dir, comet_2_polarization_dir, comet_2_merged_dir)
    make_merged_dataset(comet_3_images_dir, comet_3_polarization_dir, comet_3_merged_dir)
    make_merged_dataset(comet_4_images_dir, comet_4_polarization_dir, comet_4_merged_dir)
    make_merged_dataset(comet_5_images_dir, comet_5_polarization_dir, comet_5_merged_dir)


def rename_merged_comets():
    rename_dir_files(comet_1_merged_dir, lambda x: x + '_comet_1')
    rename_dir_files(comet_2_merged_dir, lambda x: x + '_comet_2')
    rename_dir_files(comet_3_merged_dir, lambda x: x + '_comet_3')
    rename_dir_files(comet_4_merged_dir, lambda x: x + '_comet_4')
    rename_dir_files(comet_5_merged_dir, lambda x: x + '_comet_5')


def make_merged_numbers():
    copy_number_0_images_dir = os.path.join(datasets_dir, 'number_0_images_copy')
    copy_number_0_polarization_dir = os.path.join(datasets_dir, 'number_0_polarization_copy')

    shutil.copytree(number_0_images_dir, copy_number_0_images_dir)
    shutil.copytree(number_0_polarization_dir, copy_number_0_polarization_dir)

    rename_dir_files(copy_number_0_images_dir, replace_kiril)
    rename_dir_files(copy_number_0_polarization_dir, replace_kiril)

    make_merged_dataset(copy_number_0_images_dir, copy_number_0_polarization_dir, number_0_merged_dir)
    make_merged_dataset(number_1_images_dir, number_1_polarization_dir, number_1_merged_dir)
    make_merged_dataset(number_2_images_dir, number_2_polarization_dir, number_2_merged_dir)
    make_merged_dataset(number_3_images_dir, number_3_polarization_dir, number_3_merged_dir)


def replace_kiril(string: str) -> str:
    string = string.replace('л', 'l')
    string = string.replace('б', 'b')
    return string


def rename_merged_numbers():
    rename_dir_files(number_0_merged_dir, lambda x: x + '_number_0')
    rename_dir_files(number_1_merged_dir, lambda x: x + '_number_1')
    rename_dir_files(number_2_merged_dir, lambda x: x + '_number_2')
    rename_dir_files(number_3_merged_dir, lambda x: x + '_number_3')


def extract_coco():
    annotation_paths = [
        comet_1_images_annotation_path,
        comet_2_images_annotation_path,
        comet_3_images_annotation_path,
        comet_4_images_annotation_path,
        comet_5_images_annotation_path,
        number_0_annotation_path,
        number_1_annotation_path,
        number_2_annotation_path,
        number_3_annotation_path,
    ]

    postfixes = [
        '_comet_1',
        '_comet_2',
        '_comet_3',
        '_comet_4',
        '_comet_5',
        '_number_0',
        '_number_1',
        '_number_2',
        '_number_3',
    ]

    extractor = CocoExtractor()
    full_data = {'classes': [], 'annotations': {}}
    for i, path in enumerate(annotation_paths):
        data = extractor(path)
        data = rename_annotation_files(data, lambda x: x + postfixes[i], '.jpg')
        full_data['classes'] = data['classes']
        full_data['annotations'].update(data['annotations'])

    return full_data


def install(annotations_data: dict):
    le = LabelEditor()
    changes = {
        0: 0,
        1: None,
        2: 2,
        3: 3,
        4: None,
        5: None,
        6: None,
        7: None,
    }
    classes = ['comet', 'other', 'joint', 'number', 'tube', 'sink', 'birdhouse', 'print']
    annotations_data = le.change_classes(annotations_data, changes, classes)

    installer = Installer()
    installer.install_with_splitting(annotations_data,
                                     os.path.join(datasets_dir, 'tmk_06_09_2022_raw'),
                                     os.path.join(datasets_dir, 'tmk_06_09_2022'))


if __name__ == '__main__':
    # make_merged_comets()
    # rename_merged_comets()

    # make_merged_numbers()
    # rename_merged_numbers()

    data = extract_coco()
    print(data['annotations'].keys())
    install(data)

    pass





