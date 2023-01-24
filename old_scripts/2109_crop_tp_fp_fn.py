import os
import cv2
import torch
import matplotlib.pyplot as plt

# from detection.metrics_tools import *
from detection.inference_tools.saving import detect_and_save
from detection.inference_tools.yolov5_detector import Yolov5Detector


dataset_dir = r'F:\datasets\tmk_yolov5_17092022'

train_images_dir = os.path.join(dataset_dir, 'train', 'images')
train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')

valid_images_dir = os.path.join(dataset_dir, 'valid', 'images')
valid_labels_dir = os.path.join(dataset_dir, 'valid', 'labels')

test_images_dir = os.path.join(dataset_dir, 'test', 'images')
test_labels_dir = os.path.join(dataset_dir, 'test', 'labels')

save_valid_dir = r'F:\datasets\1709_valid_detects'
save_test_dir = r'F:\datasets\1709_test_detects'

save_detects_dir = r'F:\datasets\1709_detects'

save_crop_dir = r'F:\datasets\1709_det_crops'

model_path = r'C:\Users\Alex\Downloads\yolov5l_250ep_17092022-20220920T205028Z-001\yolov5l_250ep_17092022\weights\best.pt'


def save_det_crops(images_dir, gt_dir, det_dir, save_dir):
    gt_bboxes = get_bounding_boxes(gt_dir, BBType.GroundTruth)
    det_bboxes = get_bounding_boxes(det_dir, BBType.Detected)

    evaluator = CustomEvaluator()
    metrics = evaluator.GetPascalVOCMetrics(gt_bboxes + det_bboxes, IOUThreshold=0.2)
    print(metrics[0]['AP'])
    print('tp', len(metrics[0]['tp_list']))
    print('fp', len(metrics[0]['fp_list']))
    print('gt', len(metrics[0]['gt_list']))
    print('fn', len(metrics[0]['fn_list']))

    print(metrics[0]['total positives'])

    data_types = ['tp_list', 'fp_list', 'gt_list', 'fn_list']
    for data_type in data_types:

        os.makedirs(os.path.join(save_dir, data_type), exist_ok=True)

        bb_list = metrics[0][data_type]

        for bb in bb_list:
            image_name = bb.getImageName()
            cls_id = bb.getClassId()
            
            x, y, w, h = bb.getAbsoluteBoundingBox()
            img = cv2.imread(os.path.join(images_dir, image_name + '.jpg'))
            img_crop = img[y:y + h, x:x + w]

            save_name = f'{image_name}__{x}_{y}_{w}_{h}.jpg'
            print(save_name)
            cv2.imwrite(os.path.join(save_dir, data_type, save_name), img_crop)


if __name__ == '__main__':

    detector = Yolov5Detector(model_path)
    
    for split in ['train', 'test', 'valid']:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        save_dir = os.path.join(save_detects_dir, split)
        detect_and_save(images_dir, save_dir, detector, conf=0.2)

    # detect_and_save(valid_images_dir, save_valid_dir, detector, conf=0.1)
    # detect_and_save(test_images_dir, save_test_dir, detector, conf=0.3)

    # save_det_crops(valid_images_dir, valid_labels_dir, save_valid_dir, save_crop_dir)
    # save_det_crops(test_images_dir, test_labels_dir, save_test_dir, save_crop_dir)
    



    # gt_bboxes = get_bounding_boxes(valid_labels_dir, BBType.GroundTruth)
    # det_bboxes = get_bounding_boxes(save_valid_dir, BBType.Detected)

    # evaluator = CustomEvaluator()
    # thresholds = [i / 100 for i in range(5, 100, 5)]
    # map_list = []
    # for t in thresholds:
    #     print(t)
    #     metrics = evaluator.GetPascalVOCMetrics(gt_bboxes + det_bboxes, IOUThreshold=t)
    #     map_list.append(metrics[0]['AP'])
    
    # plt.plot(thresholds, map_list)
    # plt.xlabel('IoU threshold')
    # plt.ylabel('mAP')
    # plt.grid()
    # plt.show()



