import os
import sys
from glob import glob
import cv2
import numpy as np
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.detection.dataset.detection_dataset import DetectionDataset
from cvml.detection.dataset.extractor import AnnotationExtractor, Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.image_transforming import expo
from cvml.detection.dataset.image_sources import convert_paths_to_sources

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


result_dir = r'D:\datasets\tmk_yolov5_01102022_tmp'


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
    final_dataset = DetectionDataset()
    extractor = AnnotationExtractor()
    converter = AnnotationConverter()

    for i, dataset_dir in enumerate(dataset_dirs):
        print(dataset_dir)

        dataset = DetectionDataset()

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

        annotation_data = converter.read_coco(annotation_path)
        dataset.update(image_sources, annotation_data)
        dataset.rename(renamer)    
        final_dataset = final_dataset + dataset
        print(len(final_dataset))
    
    final_dataset.split_by_dataset(r'D:\datasets\tmk_yolov5_21092022')
    print(final_dataset.splits)
    # dataset.splits['train'] = []
    final_dataset.install(result_dir, install_images=False)


