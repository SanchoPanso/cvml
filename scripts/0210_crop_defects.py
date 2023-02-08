import os
import glob
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.annotation.bounding_box import BoundingBox
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.dataset.image_transforming import expo

crop_counter = 1

yolo_dataset_dir = '/home/student2/datasets/tmk_cvs3_yolov5_02102022'
tmk_dir = '/home/student2/datasets/TMK_CVS3'
defect_dirs = {
    'tmk_1': os.path.join(tmk_dir, 'SVC3_defects TMK1'),
    'tmk_2': os.path.join(tmk_dir, 'SVC3_defects TMK2'),
    'tmk_3': os.path.join(tmk_dir, 'SVC3_defects TMK3'),
    'tmk_4': os.path.join(tmk_dir, 'SVC3_defects TMK4'),
    'tmk_5': os.path.join(tmk_dir, 'SVC3_defects TMK5'),
}


def get_masked_img(bbox: BoundingBox, final_img: np.ndarray, mask: np.ndarray, defect_id: int) -> np.ndarray:
    xc, yc, w, h = bbox.get_relative_bounding_box()
    cls_id = bbox.get_class_id()
    
    if int(cls_id) != defect_id:
        return None
    
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


def crop_defects(images_dir: str, 
                 polarization_dir: str,  
                 annotation_path: str, 
                 defect_id: int, 
                 data_mark: str,
                 save_dir: str):
    
    global crop_counter

    converter = AnnotationConverter()
    annotation_data = converter.read_coco(annotation_path)
    os.makedirs(save_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(images_dir, '*'))
    color_mask_files = glob.glob(os.path.join(images_dir, '*color_mask*'))
    img_files = list(set(all_files) - set(color_mask_files))
    
    for img_file in img_files:
        img_name, ext = os.path.splitext(os.path.split(img_file)[-1])
        if img_name not in annotation_data.bounding_boxes.keys():
            print(img_name, 'is absent')
            continue

        if not os.path.exists(os.path.join(yolo_dataset_dir, 'train', 'images', f'{img_name}_{data_mark}.jpg')):
            print(img_name, 'not in train')
            continue

        print(img_name)

        orig_img = cv2.imread(os.path.join(images_dir, img_name + ext))
        polar_1_img = cv2.imread(os.path.join(polarization_dir, img_name + '_1' + ext))
        polar_2_img = cv2.imread(os.path.join(polarization_dir, img_name + '_2' + ext))

        orig_img = expo(orig_img, 15)
        
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        polar_1_img = cv2.cvtColor(polar_1_img, cv2.COLOR_BGR2GRAY)
        polar_2_img = cv2.cvtColor(polar_2_img, cv2.COLOR_BGR2GRAY)

        mask_img = cv2.imread(os.path.join(images_dir, img_name + '_color_mask' + ext))
        if mask_img is None:
            continue
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        final_img = cv2.merge([polar_1_img, polar_2_img, orig_img])
        ret, mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

        for bbox in annotation_data.bounding_boxes[img_name]:
            masked_img = get_masked_img(bbox, final_img, mask, defect_id)
            if masked_img is not None:
                cv2.imwrite(os.path.join(save_dir, f'{crop_counter}.jpg'), masked_img)
                print(crop_counter)
                crop_counter += 1
        

if __name__ == '__main__':
    save_dir = '/home/student2/datasets/0210_defect_crops'
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
    defect_ids = [5, 8, 9, 10]

    for defect_id in defect_ids:
        crop_counter = 1
        for key in defect_dirs.keys():
            defect_dir = defect_dirs[key]
            images_dir = os.path.join(defect_dir, 'images')
            polarization_dir = os.path.join(defect_dir, 'polar')
            annotation_path = os.path.join(defect_dir, 'annotations', 'instances_default.json')
            crop_defects(images_dir, polarization_dir, annotation_path, defect_id, key, os.path.join(save_dir, class_names[defect_id]))





