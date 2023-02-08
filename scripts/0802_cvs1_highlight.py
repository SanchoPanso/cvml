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


def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*cvs1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*csv1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*number*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*comet*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))

    raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_comet_december3')))
    raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_january_Marina1')))
    

    for dataset_dir in raw_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        new_annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_with_highlights.json')

        annotation = AnnotationConverter.read_coco(annotation_path)
        annotation.classes.append('highlight')
        highlight_id = len(annotation.classes) - 1
        
        for image_path in image_files:
            name = os.path.splitext(os.path.split(image_path)[-1])[0]       
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            ret, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 250, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
            
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            counter = 0

            if name not in annotation.bbox_map:
                continue

            # for orig_bbox in annotation.bbox_map[name]:
            #     if annotation.classes[orig_bbox.get_class_id()] not in ['comet', 'number']:
            #         continue
            #     x, y, w, h = map(int, orig_bbox.get_coordinates())
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 70:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = BoundingBox(highlight_id, x, y, w, h, 1.0, name, img_size=img.shape[1:])
                
                iou_max = 0
                for orig_bbox in annotation.bbox_map[name]:
                    if annotation.classes[orig_bbox.get_class_id()] not in ['comet', 'number']:
                        continue

                    iou_max = max(iou_max, 
                                  bb_intersection_over_union(
                                      orig_bbox.get_coordinates(format=BBFormat.XYX2Y2),
                                      bbox.get_coordinates(format=BBFormat.XYX2Y2))
                    )
                
                if iou_max > 0.1:
                    continue

                annotation.bbox_map[name].append(bbox)
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
                counter += 1

            # print(counter)
            
            # cv2.imshow('img', cv2.resize(img, (400, 400)))
            # cv2.imshow('mask', cv2.resize(mask, (400, 400)))
            # cv2.waitKey()

        AnnotationConverter.write_coco(annotation, new_annotation_path, '.png')



if __name__ == '__main__':
    main()