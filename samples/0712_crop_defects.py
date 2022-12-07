import os
import glob
import sys
import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.core.bounding_box import BoundingBox
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.image_transforming import expo
from cvml.detection.augmentation.sp_estimator import SPEstimator

def main():
    # Yolo dataset dir for extracting splits
    yolo_dataset_path = '/home/student2/datasets/tmk_cvs3_yolov5_02102022'
    
    # Dir with tmk datasets for extracting bbox, mask
    tmk_dirs_path = '/home/student2/datasets/TMK_CVS3'

    save_dir = '/home/student2/datasets/crops/0611_comet_crops'
    class_names = [
        'comet', 
        'other', 
        'joint', 
        'number', 
        'tube', 
        'sink', 
        'birdhouse', 
        'print', 
        'riska', 
        'deformation defect', 
        'continuity violation'
    ]
    defect_ids = [0]                            # ids for cropping
    crop_counter = {id: 1 for id in defect_ids} # counter for naming

    # list of path to tmk dirs
    tmk_dirs = os.listdir(tmk_dirs_path)

    for tmk_dir in tmk_dirs:
        tmk_dir_path = os.path.join(tmk_dirs, tmk_dir)

        crop_counter = crop_defects(tmk_dir_path,
                                    yolo_dataset_path,
                                    defect_ids,
                                    crop_counter,
                                    save_dir)


def crop_defects(tmk_dir_path: str,
                 yolo_dataset_path: str,
                 defect_ids: list,
                 crop_counters: dict,
                 save_dir: str) -> dict:

    tmk_dirs_path = os.path.dirname(tmk_dir_path)
    tmk_dir = os.path.split(tmk_dir_path)[-1]

    all_images_paths = glob.glob(os.path.join(tmk_dirs_path, tmk_dir, 'images', '*'))
    color_mask_paths = glob.glob(os.path.join(tmk_dirs_path, tmk_dir, 'images', '*color_mask*'))
    image_paths = list(set(all_images_paths) - set(color_mask_paths))
    annotation_path = os.path.join(tmk_dirs_path, tmk_dir, 'annotations', 'instances_default.json')

    converter = AnnotationConverter()
    annotation_data = converter.read_coco(annotation_path)

    # Run through every image in tmk_dir
    for img_path in image_paths:

        # Checking img name in annotation data
        img_name, ext = os.path.splitext(os.path.split(img_path)[-1])
        if img_name not in annotation_data.bounding_boxes.keys():
            print(img_name, 'is absent')
            continue
        
        # Checking color mask in common list of masks
        color_mask_path = os.path.join(tmk_dir_path, 'images', img_path + '_color_mask.png')
        if not os.path.exists(color_mask_path):
            print(img_name, 'color mask is absent')

        # Checking img name in train set
        if not os.path.exists(os.path.join(yolo_dataset_path, 'train', 'images', f'{img_name}_{tmk_dir}.jpg')):
            print(img_name, 'not in train')
            continue

        print(img_name)

        # get preprocessed image
        img = cv2.imread(img_path)
        mixed_img = convert_to_mixed(img)

        # get binary mask
        mask_img = cv2.imread(color_mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

        # Run through each bbox in the image
        for bbox in annotation_data.bounding_boxes[img_name]:
            
            # Check class id
            class_id = bbox.get_class_id()
            if class_id not in defect_ids:
                continue
            
            # Check masked img
            masked_img = get_masked_img(bbox, mixed_img, mask)
            if masked_img is None:
                continue
            
            # Get appropriate crop counter for the class id and increment 
            crop_counter = crop_counters[class_id]
            crop_counters[class_id] += 1

            # Save crop in save_dir
            cv2.imwrite(os.path.join(save_dir, f'{crop_counter}.jpg'), masked_img)
            print(crop_counter)


def get_masked_img(bbox: BoundingBox, final_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xc, yc, w, h = bbox.get_relative_bounding_box()
    cls_id = bbox.get_class_id()
    
    height, width = final_img.shape[0:2]

    xc *= width
    yc *= height
    w *= width
    h *= height

    xc, yc, w, h = map(int, [xc, yc, w, h])
    padding_x = w // 4
    padding_y = h // 4

    x1 = max(0, xc - w//2 - padding_x)
    x2 = min(width - 1, xc + w//2 + padding_x)
    y1 = max(0, yc - h//2 - padding_y)
    y2 = min(height - 1, yc + h//2 + padding_y)

    final_img_crop = final_img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    masked_img = cv2.bitwise_and(final_img_crop, final_img_crop, mask=mask_crop)
    return masked_img


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


if __name__ == '__main__':
    main()







