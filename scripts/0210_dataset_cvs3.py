import os
import sys
from glob import glob
import cv2
import numpy as np
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.dataset.detection_dataset import DetectionDataset, LabeledImage
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.dataset.image_transforming import expo
from cvml.dataset.image_source import convert_paths_to_sources

source_dir = '/home/student2/datasets/TMK_CVS3'

comet_1_dir = os.path.join(source_dir, 'csv1_comets_1_24_08_2022')
comet_2_dir = os.path.join(source_dir, 'csv1_comets_2_24_08_2022')
comet_3_dir = os.path.join(source_dir, 'csv1_comets_23_08_2022')
comet_4_dir = os.path.join(source_dir, 'csv1_comets_01_09_2022')
comet_5_dir = os.path.join(source_dir, 'csv1_comets_05_09_2022')


dataset_dirs = {
    'tmk_1': os.path.join(source_dir, 'SVC3_defects TMK1'),
    'tmk_2': os.path.join(source_dir, 'SVC3_defects TMK2'),
    'tmk_3': os.path.join(source_dir, 'SVC3_defects TMK3'),
    'tmk_4': os.path.join(source_dir, 'SVC3_defects TMK4'),
    'tmk_5': os.path.join(source_dir, 'SVC3_defects TMK5'),
}

renamers = {}
for key in dataset_dirs.keys():
    renamers[key] = lambda x: x + '_' + key


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

    converter = AnnotationConverter()
    editor = AnnotationEditor()
    final_dataset = DetectionDataset()

    for key in dataset_dirs.keys():
        
        dataset = DetectionDataset()

        dataset_dir = dataset_dirs[key]
        print(dataset_dir)

        image_dir = os.path.join(dataset_dir, 'images')
        polarization_dir = os.path.join(dataset_dir, 'polar')

        channel_0_paths = glob(os.path.join(polarization_dir, '*_1.*'))
        channel_1_paths = glob(os.path.join(polarization_dir, '*_2.*'))
        channel_2_paths = list(set(glob(os.path.join(image_dir, '*'))) - set(glob(os.path.join(image_dir, '*color*'))))

        channel_0_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))
        channel_1_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))
        channel_2_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('_')[0]))

        image_sources = convert_paths_to_sources(paths=[channel_0_paths, channel_1_paths, channel_2_paths],
                                                 main_channel=2,
                                                 preprocess_fns=[None, None, wrap_expo])

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = renamers[key]

        annotation_data = converter.read_coco(annotation_path)

        changes = {
            0: None,    # comet 
            1: None,       # other
            2: None,    # joint 
            3: None,    # number
            #4: 4,       # tube
            #5: 5,       # sink
            6: None,    # birdhouse
            7: None,    # print
            #8: 8,       # riska
            #9: 9,       # deformation defect
            #10: 10,     # continuity violation
        }
        annotation_data = editor.change_classes_by_id(annotation_data, changes)

        dataset.update(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    result_dir = '/home/student2/datasets/tmk_cvs3_yolov5_02102022'
    # final_dataset.split_by_proportions({'train': 0.7, 'valid': 0.2, 'test': 0.1})
    final_dataset.split_by_dataset(result_dir)
    for idx in final_dataset.samples['valid']:
        lbl_img = final_dataset[idx]
        bboxes = lbl_img.bboxes
        name = lbl_img.name
        for bb in bboxes:
            if bb.get_class_id() == 10:
                print(name)
    
    # final_dataset.install(result_dir)


