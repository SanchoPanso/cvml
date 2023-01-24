import os
import sys
import cv2
import glob
from pathlib import Path
import numpy as np
from typing import List

sys.path.append(os.path.dirname(__file__) + '/..')

from cvml.instance_segmentation.dataset.image_transforming import get_mask_contours
from cvml.detection.dataset.annotation_converter import Annotation, AnnotationConverter
from cvml.core.bounding_box import BoundingBox


def main():
    # common_dir = Path(r'C:\Users\HP\Downloads\cvs1_comet_january1')
    # dataset_dirs = [x for x in common_dir.iterdir() if x.is_dir()]
    
    dataset_dirs = [Path(r'C:\Users\HP\Downloads\number_december1')]
    
    for dataset_dir in dataset_dirs:
        images_dir = dataset_dir / 'images'
        
        orig_annotation_path = dataset_dir / 'annotations' / 'instances_default_old.json'
        result_annotation_path = dataset_dir / 'annotations' / 'instance_segmentation.json'
        
        annotation = AnnotationConverter.read_coco(orig_annotation_path)
        masks_paths = list(images_dir.glob(f'*color_mask*'))
        for mask_path in masks_paths:
            
            name = '_'.join(mask_path.name.split('_')[:-2])
            if name not in annotation.bbox_map:
                continue
            
            mask = cv2.imread(str(mask_path))
            
            print(mask_path)
            
            bboxes = annotation.bbox_map[name]
            for bbox in bboxes:
                if bbox.get_class_id() == 3:
                    continue
                polygon = mask_to_polygons(mask, bbox)
                bbox._segmentation = polygon ## CHECK

        AnnotationConverter.write_coco(annotation, str(result_annotation_path), '.png')


def mask_to_polygons(mask: np.ndarray, bbox: BoundingBox) -> List[List[int]]:
    x, y, w, h = map(int, bbox.get_absolute_bounding_box())
    bbox_crop = mask[y: y + h, x: x + w]
    
    polygons = []
    contours = get_mask_contours(bbox_crop)
    for contour in contours:
        polygon = []
        for i in range(contour.shape[0]):
            xi = int(contour[i][0][0])
            yi = int(contour[i][0][1])
            polygon.append(x + xi)
            polygon.append(y + yi)
            
        polygons.append(polygon)
    
    return polygons


if __name__ == '__main__':
    main()