import os
import cv2
from extractor import Extractor

dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets'
downloads_dir = r'C:\Users\Alex\Downloads'

images1_dir = os.path.join(dataset_dir, 'defects1_merged_05082022')
images2_dir = os.path.join(dataset_dir, 'defects2_merged_05082022')

masks1_dir = os.path.join(downloads_dir, 'SCV1_segment_1')
masks2_dir = os.path.join(downloads_dir, 'SCV1_segment_2')


save_dir = os.path.join(dataset_dir, 'defects_segment_15082022')
mask_postfix = '_color_mask'
dir_postfix1 = '1'
dir_postfix2 = '2'

detection_annotation1 = os.path.join(downloads_dir, r'csv1_comet_1_12_08\annotations\instances_default.json')
detection_annotation2 = os.path.join(downloads_dir, r'csv1_comet_2_11_08\annotations\instances_default.json')
comet_id = 0


def crop_segments(images_dir, masks_dir, detection_annotation, save_dir, dir_postfix):
    extr = Extractor()
    data = extr.extract(detection_annotation)

    os.makedirs(save_dir, exist_ok=True)

    for img_file in data['annotations'].keys():
        print(img_file)
        name, ext = os.path.splitext(img_file)
        mask_file = name + mask_postfix + ext

        if os.path.exists(os.path.join(masks_dir, mask_file)) is False:
            continue

        img = cv2.imread(os.path.join(images_dir, img_file))
        mask = cv2.imread(os.path.join(masks_dir, mask_file))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        height, width = img.shape[0:2]
        for i, line in enumerate(data['annotations'][img_file]):
            cls_id, xc, yc, w, h = line

            if int(cls_id) != comet_id:
                continue

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

            # print(x1, y1, x2, y2)
            img_crop = img[y1:y2, x1:x2]
            mask_crop = mask[y1:y2, x1:x2]

            # cv2.imshow('test', img_crop)
            # cv2.waitKey(0)
            # cv2.imshow('test', mask_crop)
            # cv2.waitKey(0)

            obj_postfix = str(i + 1)
            cv2.imwrite(os.path.join(save_dir, f'{name}_{dir_postfix}_{obj_postfix}.jpg'), img_crop)
            cv2.imwrite(os.path.join(save_dir, f'{name}_{dir_postfix}_{obj_postfix}_mask.jpg'), mask_crop)


crop_segments(images1_dir, masks1_dir, detection_annotation1, save_dir, dir_postfix1)
crop_segments(images2_dir, masks2_dir, detection_annotation2, save_dir, dir_postfix2)



