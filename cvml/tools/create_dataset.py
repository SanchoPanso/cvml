import os
import sys
import glob
import cv2
import numpy as np
import torch
from typing import Callable, List
import argparse

# sys.path.append(os.path.dirname(__file__) + '/../..')

from cvml.core.bounding_box import BoundingBox, CoordinatesType, BBType, BBFormat
from cvml.detection.dataset.detection_dataset import DetectionDataset
from cvml.detection.dataset.image_source import ImageSource
from cvml.detection.dataset.annotation import Annotation
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.detection.dataset.image_source import convert_paths_to_single_sources
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.detection.dataset.image_transforming import convert_to_mixed, expo

from cvml.detection.augmentation.golf_augmentation import MaskMixup, MaskMixupAugmentation


def create_tubes_detection_dataset(
    source_dirs: List[str],
    save_dir: str,
    classes: List[str] = None,
    sample_proportions: dict = None,
    use_polar: bool = True,
    install_images: bool = True,
    install_labels: bool = True,
    install_annotations: bool = True,
    install_description: bool = True,
    mask_mixup_augmentation: MaskMixupAugmentation = None,
    augmentation_samples: List[str] = None,
    crop_obj_dir: str = None,
    crop_class_names: List[str] = None,
    create_compressed_samples: bool = True,
):
    """Create special detection dataset, where images preprocessed by polarization algorythm (convert_to_mixed),
    renamed by scheme f"{source_dataset_dir}_{source_image_name}" and converted in ".jpg" format.

    :param source_dirs: list of source dirs in special format. 
                        source_dir_name
                        |-images
                        |-annotations
                          |-instances_default.json
                        
                        images - subdir with original images
                        annotations/instances_default.json - annotaion in coco-format
    
    :param save_dir: path to dir for saving dataset
    :param classes: classes, chosen from annotation and renumbered, defaults to None (choosing all classes)
    :param sample_proportions: dict with keys - names of samples, values - indexes of images in sample, defaults to None
    :param install_images: _description_, defaults to True
    :param install_labels: _description_, defaults to True
    :param install_annotations: _description_, defaults to True
    :param install_description: _description_, defaults to True
    :param create_compressed_samples: _description_, defaults to True
    """
    
    final_dataset = DetectionDataset()
    
    for dataset_dir in source_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        preprocess_fn = convert_to_mixed if use_polar else lambda x: expo(x, 15)
        image_sources = convert_paths_to_single_sources(paths=image_files,
                                                        preprocess_fn=preprocess_fn)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = lambda x: os.path.split(dataset_dir)[-1] + '_' + x

        annotation_data = AnnotationConverter.read_coco(annotation_path)
        classes = classes or annotation_data.classes
        annotation_data = AnnotationEditor.change_classes_by_new_classes(annotation_data, classes)

        dataset = DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    sample_proportions = sample_proportions or {}
    final_dataset.split_by_proportions(sample_proportions)
    final_dataset.install(save_dir, install_images, install_labels, install_annotations, install_description)
    
    # mask_mixup_augmentation
    if mask_mixup_augmentation is not None:
        
        # Create augmenter
        augmentation_samples = augmentation_samples or ['train']
        crop_class_names = [] if crop_class_names is None else crop_class_names
        augmenter = MaskMixup(crop_obj_dir, crop_class_names, final_dataset.annotation.classes)
        
        for sample in augmentation_samples:
            
            # Create new dir for aug sample
            if sample not in final_dataset.samples:
                continue
            aug_sample = f"{sample}_aug"
            
            new_images_dir = os.path.join(save_dir, aug_sample, 'images')
            new_labels_dir = os.path.join(save_dir, aug_sample, 'labels')
            new_annotations_dir = os.path.join(save_dir, aug_sample, 'annotations')

            os.makedirs(new_images_dir, exist_ok=True)
            os.makedirs(new_labels_dir, exist_ok=True)
            os.makedirs(new_annotations_dir, exist_ok=True)

            # Create new aug dataset 
            aug_coco_sample_annotation = Annotation(final_dataset.annotation.classes, {})
            aug_yolo_sample_annotation = Annotation(final_dataset.annotation.classes, {})
            aug_image_sources = []
            for idx in final_dataset.samples[sample]:
                
                image_source = final_dataset.image_sources[idx]
                aug_image_sources.append(image_source)
                name = image_source.name
                bboxes = final_dataset.annotation.bbox_map[name]
                aug_coco_sample_annotation.bbox_map[name] = bboxes
                
                # read img and labels
                img = image_source.read()
                labels = bboxes_to_labels(bboxes, img.shape)
                
                # apply mask mixup
                result = mask_mixup_augmentation(augmenter, img, labels)
                if result is None:
                    continue
                new_img, new_labels = result
                new_name = name + '_aug'
                
                # install images separately for annotations
                if install_images:
                    imwrite(os.path.join(new_images_dir, new_name + '.jpg'), new_img)
                
                # Write new bboxes in new annotation
                new_bboxes = labels_to_bboxes(new_labels, new_name, img.shape)
                aug_coco_sample_annotation.bbox_map[new_name] = new_bboxes
                aug_yolo_sample_annotation.bbox_map[new_name] = new_bboxes
            
            if install_labels:
                AnnotationConverter.write_coco(aug_coco_sample_annotation, os.path.join(new_annotations_dir, 'data.json'))
            if install_annotations:
                AnnotationConverter.write_yolo(aug_yolo_sample_annotation, new_labels_dir)
            
    
    if create_compressed_samples:
        if os.name == 'posix':
            all_samples = list(sample_proportions.keys())
            if mask_mixup_augmentation is None:
                aug_samples = [f'{i}_aug' for i in sample_proportions.keys()]
                all_samples += aug_samples
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