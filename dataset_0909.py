import os
import cv2
import numpy as np
from detection.dataset_tools.dataset import YoloDataset
from detection.dataset_tools.coco_extractor import CocoExtractor
from typing import Callable
from detection.dataset_tools.image_transforming import expo
from detection.dataset_tools.label_editor import LabelEditor


tmk_dir = r'F:\TMK'

comet_1_dir = os.path.join(tmk_dir, 'csv1_comets_1_24_08_2022')
comet_2_dir = os.path.join(tmk_dir, 'csv1_comets_2_24_08_2022')
comet_3_dir = os.path.join(tmk_dir, 'csv1_comets_23_08_2022')
comet_4_dir = os.path.join(tmk_dir, 'csv1_comets_01_09_2022')
comet_5_dir = os.path.join(tmk_dir, 'csv1_comets_05_09_2022')

number_0_dir = os.path.join(tmk_dir, 'numbers_23_06_2021')
number_1_dir = os.path.join(tmk_dir, 'numbers_24_08_2022')
number_2_dir = os.path.join(tmk_dir, 'numbers_25_08_2022')
number_3_dir = os.path.join(tmk_dir, 'numbers_01_09_2022')


dataset_dirs = [
    comet_1_dir,
    comet_2_dir,
    comet_3_dir,
    comet_4_dir,
    comet_5_dir,
    number_0_dir,
    number_1_dir,
    number_2_dir,
    number_3_dir,
]

renamers = [
    lambda x: x + '_comet_1',
    lambda x: x + '_comet_2',
    lambda x: x + '_comet_3',
    lambda x: x + '_comet_4',
    lambda x: x + '_comet_5',

    lambda x: x + '_number_0',
    lambda x: x + '_number_1',
    lambda x: x + '_number_2',
    lambda x: x + '_number_3',
]


result_dir = r'F:\datasets\tmk_09_09_2022'


def wrap_expo(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = expo(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rename_annotation_files(annotations_data: dict,  rename_callback: Callable, new_ext: str = None) -> dict:
    filenames = list(annotations_data['annotations'].keys())
    new_annotations_data = {'classes': annotations_data['classes'], 'annotations': {}}
    for filename in filenames:
        name, ext = os.path.splitext(filename)
        new_name = rename_callback(name)

        if new_ext is None:
            new_filename = new_name + ext
        else:
            new_filename = new_name + new_ext

        labels = annotations_data['annotations'][filename]
        new_annotations_data['annotations'][new_filename] = labels

    return new_annotations_data


if __name__ == '__main__':
    dataset = YoloDataset()
    extractor = CocoExtractor()
    label_editor = LabelEditor()

    for i, dataset_dir in enumerate(dataset_dirs):
        print(dataset_dir)

        image_dir = os.path.join(dataset_dir, 'images')
        polarization_dir = os.path.join(dataset_dir, 'polarization')
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = renamers[i]

        annotation_data = extractor(annotation_path)
        annotation_data = rename_annotation_files(annotation_data, lambda x: x, '.png')
        changes = {0: 0, 1: None, 2: 2, 3: 3, 4: None, 5: None, 6: None, 7: None}
        label_editor.change_classes(annotation_data, changes, annotation_data['classes'])

        dataset.add(image_dir=image_dir,
                    polarization_dir=polarization_dir,
                    annotation_data=annotation_data,
                    rename_callback=renamer,
                    preprocessers=[None, None, wrap_expo])

    dataset.split_by_proportions({'train': 0.7, 'valid': 0.2, 'test': 0.1})
    print(dataset.splits)
    dataset.install(result_dir)


