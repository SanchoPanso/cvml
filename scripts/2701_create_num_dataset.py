import os
import sys
import glob
import cv2
import numpy as np
import torch
import logging
from typing import Callable, List
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cvml.annotation.bounding_box import BoundingBox, CoordinatesType, BBType, BBFormat
from cvml.dataset.detection_dataset import DetectionDataset
from cvml.dataset.image_source import ImageSource
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.dataset.image_transforming import convert_to_mixed, expo

from cvml.detection.augmentation.golf_augmentation import MaskMixup, MaskMixupAugmentation


# def create_tubes_detection_dataset(
#     source_dirs: List[str],
#     save_dir: str,
#     classes: List[str] = None,
#     sample_proportions: dict = None,
#     template_dataset: str = None,
#     use_polar: bool = True,
#     install_images: bool = True,
#     install_labels: bool = True,
#     install_annotations: bool = True,
#     install_description: bool = True,
#     mask_mixup_augmentation: MaskMixupAugmentation = None,
#     augmentation_samples: List[str] = None,
#     crop_obj_dir: str = None,
#     crop_class_names: List[str] = None,
#     create_compressed_samples: bool = True,
# ):
#     """Create special detection dataset, where images preprocessed by polarization algorythm (convert_to_mixed),
#     renamed by scheme f"{source_dataset_dir}_{source_image_name}" and converted in ".jpg" format.

#     :param source_dirs: list of source dirs in special format. 
#                         source_dir_name
#                         |-images
#                         |-annotations
#                           |-instances_default.json
                        
#                         images - subdir with original images
#                         annotations/instances_default.json - annotaion in coco-format
    
#     :param save_dir: path to dir for saving dataset
#     :param classes: classes, chosen from annotation and renumbered, defaults to None (choosing all classes)
#     :param sample_proportions: dict with keys - names of samples, values - indexes of images in sample, defaults to None
#     :param install_images: _description_, defaults to True
#     :param install_labels: _description_, defaults to True
#     :param install_annotations: _description_, defaults to True
#     :param install_description: _description_, defaults to True
#     :param create_compressed_samples: _description_, defaults to True
#     """

def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = []
    raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '*number1*'))
    raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '*number2*'))
    raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '*number3*'))
    raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир'))
    raw_dirs.sort()

    save_dir = '/home/student2/datasets/prepared/cvs1_number_27012023' #'/home/student2/datasets/prepared/TMK_CVS3_0701'

    create_compressed_samples = True

    cls_names = [str(i) for i in range(9)]
    split_proportions = {'train': 0.8, 'valid': 0.2, 'test': 0.0}

    cvml_logger = logging.getLogger('cvml')

    # Create handlers
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)

    # Add handlers to the logger
    cvml_logger.addHandler(s_handler)

    
    final_dataset = DetectionDataset()
    
    for dataset_dir in raw_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        preprocess_fn = lambda x: expo(x, 15)
        image_sources = convert_paths_to_single_sources(paths=image_files,
                                                        preprocess_fn=preprocess_fn)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'digits.json')
        renamer = lambda x: os.path.split(dataset_dir)[-1] + '_' + x

        annotation_data = AnnotationConverter.read_coco(annotation_path)
        classes = cls_names
        annotation_data = AnnotationEditor.change_classes_by_new_classes(annotation_data, classes)

        dataset = DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    sample_proportions = split_proportions
    final_dataset.split_by_proportions(sample_proportions)
    final_dataset.install(
        save_dir, 
        install_images=True, 
        install_labels=True, 
        install_annotations=True, 
        install_description=True
    )
    
    if create_compressed_samples:
        if os.name == 'posix':
            all_samples = list(sample_proportions.keys())

            for sample_name in all_samples:
                sample_path = os.path.join(save_dir, sample_name)
                os.system(f" cd {save_dir}; zip -r {sample_name}.zip {sample_name}/*")
                os.system(f"split {sample_path}.zip {sample_path}.zip.part_ -b 999MB")
        elif os.name == 'nt':
            os.system("echo Zip is not Implemented")    # TODO
        else:
            pass



def bboxes_to_labels(bboxes: List[BoundingBox], img_size: tuple) -> np.ndarray:
    labels_list = []
    for bbox in bboxes:
        xc, yc, w, h = bbox.get_relative_bounding_box(img_size)
        cls_id = bbox.get_class_id()
        labels_list.append([cls_id, xc, yc, w, h])
    
    if len(labels_list) == 0:
        return np.zeros((0, 5))
    return np.array(labels_list)


def labels_to_bboxes(labels: np.ndarray, image_name: str, image_size: tuple) -> List[BoundingBox]:
    bboxes = []
    for label in labels:
        cls_id, xc, yc, w, h = label
        bbox = BoundingBox(cls_id, xc, yc, w, h, 1.0, 
                           image_name, 
                           CoordinatesType.Relative, 
                           image_size, 
                           BBType.GroundTruth, 
                           BBFormat.XYWH)
        bboxes.append(bbox)
    return bboxes
    
    
def imwrite(path: str, img: np.ndarray):
    ext = os.path.splitext(os.path.split(path)[-1])[1]
    is_success, im_buf_arr = cv2.imencode(ext, img)
    im_buf_arr.tofile(path)


if __name__ == '__main__':
    main()