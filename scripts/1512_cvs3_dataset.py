import os
import sys
import glob
import cv2
import numpy as np
import torch
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.dataset.detection_dataset import DetectionDataset, LabeledImage
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.dataset.image_transforming import expo
from cvml.dataset.image_source import convert_single_paths_to_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator

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
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = expo(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_min_max(data):
    data_min = data.min()
    data_max = data.max()
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data


def convert_to_mixed(orig_img: np.ndarray) -> np.ndarray:

    height, width = orig_img.shape[0:2]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    in_data = torch.from_numpy(img).float() #torch.frombuffer(orig_img.data, dtype=torch.uint8, count=img.size).float().detach_().reshape(height, width)
    
    estimator = SPEstimator()
    rho, phi = estimator.getAzimuthAndPolarization(in_data)
    
    normalized_rho = normalize_min_max(rho)
    normalized_phi = normalize_min_max(phi)

    rho_img = (normalized_rho * 255).numpy().astype('uint8')
    phi_img = (normalized_phi * 255).numpy().astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray_img = expo(img, 15)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    img = cv2.merge([phi_img, rho_img, gray_img])
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

    raw_datasets_dir = '/home/student2/datasets/raw/TMK_3010'
    raw_dirs = glob.glob(os.path.join(raw_datasets_dir, '*SCV3*'))
    raw_dirs.sort()

    for dataset_dir in raw_dirs:
        
        dataset = DetectionDataset()

        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        image_sources = convert_single_paths_to_sources(paths=image_files,
                                                        preprocess_fn=convert_to_mixed)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = lambda x: x + '_' + os.path.split(dataset_dir)[-1]

        annotation_data = converter.read_coco(annotation_path)

        changes = {
            0: None,    # comet 
            1: 1,       # other
            2: None,    # joint 
            3: None,    # number
            4: 4,       # tube
            5: 5,       # sink
            6: None,    # birdhouse
            7: None,    # print
            8: 8,       # riska
            9: None,       # deformation defect
            10: None,     # continuity violation
        }
        annotation_data = editor.change_classes_by_id(annotation_data, changes)

        dataset.update(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    result_dir = '/home/student2/datasets/prepared/tmk_cvs3_yolov5_17122022'
    final_dataset.split_by_proportions({'train': 0.8, 'valid': 0.2, 'test': 0.0})

    print(len(final_dataset.samples['train']))
    print(len(final_dataset.samples['valid']))
    print(len(final_dataset.samples['test']))

    valid_cnt = {}
    for i in final_dataset.samples['valid']:
        lbl_img = final_dataset.labeled_images[i]
        for bb in lbl_img.bboxes:
            cls_id = bb.get_class_id() 
            if cls_id in valid_cnt:
                valid_cnt[cls_id] += 1
            else:
                valid_cnt[cls_id] = 1

    train_cnt = {}
    for i in final_dataset.samples['train']:
        lbl_img = final_dataset.labeled_images[i]
        for bb in lbl_img.bboxes:
            cls_id = bb.get_class_id() 
            if cls_id in train_cnt:
                train_cnt[cls_id] += 1
            else:
                train_cnt[cls_id] = 1
    
    final_dataset.install(result_dir)


