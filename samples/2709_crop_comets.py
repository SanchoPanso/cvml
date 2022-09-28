import os
import sys
import cv2
import numpy as np
from typing import Set

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_science_tools.detection_collection.dataset_tools.extractor import Extractor
from data_science_tools.detection_collection.dataset_tools.image_transforming import expo

crop_counter = 1
comet_id = 0

tmk_dir = r'F:\TMK'
comet_dirs = {
    'comet_3': os.path.join(tmk_dir, 'csv1_comets_23_08_2022'),
    'comet_4': os.path.join(tmk_dir, 'csv1_comets_01_09_2022'),
    'comet_5': os.path.join(tmk_dir, 'csv1_comets_05_09_2022'),
}


def get_masked_img(label: list, final_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cls_id, xc, yc, w, h = label
    
    if int(cls_id) != comet_id:
        return None
    
    height, width = final_img.shape[0:2]

    xc *= width
    yc *= height
    w *= width
    h *= height

    xc, yc, w, h = map(int, [xc, yc, w, h])
    padding_x = w // 3
    padding_y = h // 3

    x1 = max(0, xc - w//2 - padding_x)
    x2 = min(width - 1, xc + w//2 + padding_x)
    y1 = max(0, yc - h//2 - padding_y)
    y2 = min(height - 1, yc + h//2 + padding_y)

    final_img_crop = final_img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    masked_img = cv2.bitwise_and(final_img_crop, final_img_crop, mask=mask_crop)
    return masked_img


def get_train_names(train_dir) -> Set[str]:
    files = os.listdir(train_dir)
    names = set()
    for file in files:
        name, ext = os.path.splitext(file)
        names.add(name)
    return names


def crop_comets(images_dir, polarization_dir, color_masks_dir, annotation_path, 
                save_dir, postfix: str, train_names: set):
    
    global crop_counter

    extractor = Extractor()
    annotation_data = extractor(annotation_path)
    os.makedirs(save_dir, exist_ok=True)
    
    img_files = os.listdir(images_dir)
    for img_file in img_files:
        img_name, ext = os.path.splitext(img_file)

        if f'{img_name}_{postfix}' not in train_names:
            print(img_name, 'is not in train')
            continue

        if img_name not in annotation_data['annotations'].keys():
            print(img_name, 'is absent')
            continue
        print(img_name)

        orig_img = cv2.imread(os.path.join(images_dir, img_name + ext))
        polar_1_img = cv2.imread(os.path.join(polarization_dir, img_name + '_1' + ext))
        polar_2_img = cv2.imread(os.path.join(polarization_dir, img_name + '_2' + ext))

        orig_img = expo(orig_img, 15)
        
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        polar_1_img = cv2.cvtColor(polar_1_img, cv2.COLOR_BGR2GRAY)
        polar_2_img = cv2.cvtColor(polar_2_img, cv2.COLOR_BGR2GRAY)

        mask_img = cv2.imread(os.path.join(color_masks_dir, img_name + '_color_mask' + ext))
        if mask_img is None:
            continue
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        final_img = cv2.merge([polar_1_img, polar_2_img, orig_img])
        ret, mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

        for label in annotation_data['annotations'][img_name]:
            masked_img = get_masked_img(label, final_img, mask)
            if masked_img is not None:
                cv2.imwrite(os.path.join(save_dir, f'{crop_counter}.jpg'), masked_img)
                print(crop_counter)
                crop_counter += 1


if __name__ == '__main__':
    save_dir = 'F:\\datasets\\2709_comet_crops'

    train_names = get_train_names(r'F:\datasets\tmk_yolov5_25092022\train\images')
    print(train_names)

    for key in comet_dirs.keys():
        comet_dir = comet_dirs[key]
        images_dir = os.path.join(comet_dir, 'images')
        polarization_dir = os.path.join(comet_dir, 'polarization')
        color_masks_dir = os.path.join(comet_dir, 'color_masks')
        annotation_path = os.path.join(comet_dir, 'annotations', 'instances_default.json')
        crop_comets(images_dir, polarization_dir, color_masks_dir, annotation_path, save_dir, key, train_names)





