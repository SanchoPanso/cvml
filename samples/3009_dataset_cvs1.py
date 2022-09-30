import os
import sys
from glob import glob
import cv2
import numpy as np
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_science_tools.detection_collection.dataset_tools.dataset import Coco2YoloDataset
from data_science_tools.detection_collection.dataset_tools.extractor import Extractor
from data_science_tools.detection_collection.dataset_tools.image_transforming import expo
from data_science_tools.detection_collection.dataset_tools.label_editor import LabelEditor
from data_science_tools.detection_collection.dataset_tools.image_sources import convert_paths_to_sources

tmk_dir = r'D:\TMK'

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


result_dir = r'D:\datasets\tmk_yolov5_30092022_tmp'


def wrap_expo(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = expo(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rename_annotation_files(annotations_data: dict,  rename_callback: Callable) -> dict:
    names = list(annotations_data['annotations'].keys())
    new_annotations_data = {'classes': annotations_data['classes'], 'annotations': {}}
    for name in names:
        new_name = rename_callback(name)

        labels = annotations_data['annotations'][name]
        new_annotations_data['annotations'][new_name] = labels

    return new_annotations_data


if __name__ == '__main__':
    dataset = Coco2YoloDataset()
    extractor = Extractor()
    label_editor = LabelEditor()

    for i, dataset_dir in enumerate(dataset_dirs):
        print(dataset_dir)

        image_dir = os.path.join(dataset_dir, 'images')
        polarization_dir = os.path.join(dataset_dir, 'polarization')

        channel_0_paths = glob(os.path.join(polarization_dir, '*_1.*'))
        channel_1_paths = glob(os.path.join(polarization_dir, '*_2.*'))
        channel_2_paths = glob(os.path.join(image_dir, '*'))

        channel_0_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))
        channel_1_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))
        channel_2_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))

        image_sources = convert_paths_to_sources(paths=[channel_0_paths, channel_1_paths, channel_2_paths],
                                                 main_channel=2,
                                                 preprocess_fns=[None, None, wrap_expo])

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = renamers[i]

        annotation_data = extractor(annotation_path)
        annotation_data = rename_annotation_files(annotation_data, lambda x: x)
        changes = {0: 0, 
                   1: 1, 
                   2: 2, 
                   3: 3, 
                   4: 4, 
                   5: 5, 
                   6: 6, 
                   7: 7}
        label_editor.change_classes(annotation_data, changes, annotation_data['classes'])

        dataset.add(image_sources=image_sources,
                    annotation_data=annotation_data,
                    rename_callback=renamer)

    # dataset.split_by_proportions({'train': 0.7, 'valid': 0.2, 'test': 0.1})
    dataset.split_by_dataset(r'D:\datasets\tmk_yolov5_21092022')
    print(dataset.splits)
    # dataset.splits['train'] = []
    dataset.install(result_dir, install_images=False)


