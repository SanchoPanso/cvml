import os
import cv2
from typing import List

from detection.inference_tools.visualizer import Visualiser
from detection.metrics_tools import *
from detection.dataset_tools.io_handling import read_yolo_labels


IMG_SIZE = (2448, 2048)


def get_bboxes_from_file(img_size: tuple, txt_file: str, bb_type: BBType) -> CustomBoundingBoxes:
    img_name, ext = os.path.splitext(txt_file)
    labels = read_yolo_labels(txt_file)
    bounding_boxes = CustomBoundingBoxes()

    for line in labels:
        if bb_type == BBType.GroundTruth:
            cls_id, x, y, w, h = line[:5]
            bbox = BoundingBox(img_name, cls_id,
                               x, y, w, h,
                               CoordinatesType.Relative, img_size,
                               bb_type,
                               format=BBFormat.XYWH)
        else:
            cls_id, x, y, w, h, cls_conf = line
            bbox = BoundingBox(img_name, cls_id,
                               x, y, w, h,
                               CoordinatesType.Relative, img_size,
                               bb_type,
                               cls_conf,
                               format=BBFormat.XYWH)

        bounding_boxes.addBoundingBox(bbox)
    return bounding_boxes
    

def get_fp_bboxes(det_bboxes: CustomBoundingBoxes, 
                  gt_bboxes: CustomBoundingBoxes,
                  IOUThreshold: float = 0.2,
                  class_id: int = 0) -> List[BoundingBox]:
    
    evaluator = CustomEvaluator()
    metrics = evaluator.GetPascalVOCMetrics(gt_bboxes + det_bboxes, IOUThreshold)

    if len(metrics) == 0:
        return []
    
    return metrics[class_id]['fp_list']


def get_save_path(name: str) -> str:
    parts = name.split('_')

    img_num = parts[0]
    data_num = '_'.join(parts[1:])

    path_dict = {
        'comet_1': 'comet_1',
        'comet_2': 'comet_2',
        'comet_3': 'comet_3',
        'comet_4': 'comet_4',
        'comet_5': 'comet_5',
    }

    for key in path_dict.keys():
        if data_num == key:
            return os.path.join(path_dict[key], img_num + '.png')
    
    return None


def make_fp_vis(det_dir: str, gt_dir: str, images_dir: str, save_dir: str):
    
    vis = Visualiser()
    gt_txt_files = os.listdir(gt_dir)
    
    for txt_file in gt_txt_files:
        if not os.path.exists(os.path.join(det_dir, txt_file)):
            continue

        gt_bboxes = get_bboxes_from_file(IMG_SIZE, os.path.join(gt_dir, txt_file), BBType.GroundTruth)
        det_bboxes = get_bboxes_from_file(IMG_SIZE, os.path.join(det_dir, txt_file), BBType.Detected)

        fp_bboxes = get_fp_bboxes(det_bboxes, gt_bboxes)
        tmp_fp_bboxes = []
        for fp_bb in fp_bboxes:
            cls_id = fp_bb.getClassId()
            if cls_id == 0:
                tmp_fp_bboxes.append(fp_bb)
            else:
                print('wrong class', cls_id)
        fp_bboxes = tmp_fp_bboxes

        if len(fp_bboxes) == 0:
            print(txt_file, 'fp none')
            continue
        
        name, ext = os.path.splitext(txt_file)
        img = cv2.imread(os.path.join(images_dir, name + '.jpg'))
        img = cv2.merge([cv2.split(img)[2]] * 3)

        for fp_bb in fp_bboxes:
            x, y, w, h = fp_bb.getAbsoluteBoundingBox()
            conf = fp_bb.getConfidence()
            color = (0, 0, 255)
            description = 'comet {0:.3f}'.format(conf)
            vis.draw_bbox(img, x, y, w, h, color, description)
        
        save_path = get_save_path(name)

        if save_path is None:
            print(txt_file, 'save none')
            continue

        os.makedirs(os.path.dirname(os.path.join(save_dir, save_path)), exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, save_path), img)
        print(txt_file, 'ok')


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

    data_type = 'fp_list'
    os.makedirs(os.path.join(save_dir), exist_ok=True)

    bb_list = metrics[0][data_type]
    bb_dict = {}
    for bb in bb_list:
        image_name = bb.getImageName()
        if image_name in bb_dict.keys():
            bb_dict[image_name].append(bb)
        else:
            bb_dict[image_name] = [bb]
        
    vis = Visualiser()
    for image_name in bb_dict.keys():
        
        img = cv2.imread(os.path.join(images_dir, image_name + '.jpg'))
        img = cv2.merge([cv2.split(img)[2]] * 3)

        for bb in bb_dict[image_name]:
            
            cls_id = bb.getClassId()
            
            x, y, w, h = bb.getAbsoluteBoundingBox()
            conf = bb.getConfidence()
            color = (0, 0, 255)
            description = 'comet {0:.3f}'.format(conf)
            vis.draw_bbox(img, x, y, w, h, color, description)

        name, ext = os.path.splitext(image_name)
        save_path = get_save_path(name)
        if save_path is None:
            print(name, 'save none')
            continue

        os.makedirs(os.path.dirname(os.path.join(save_dir, save_path)), exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, save_path), img)
        print(save_path)



if __name__ == '__main__':
    save_dir = r'F:\datasets\1709_fp'
    detects_dir = r'F:\datasets\1709_detects'
    dataset_dir = r'F:\datasets\tmk_yolov5_21092022'

    for split in ['valid', 'test', 'train']:
        gt_dir = os.path.join(dataset_dir, split, 'labels')
        det_dir = os.path.join(detects_dir, split)
        images_dir = os.path.join(r'F:\datasets\tmk_yolov5_17092022', split, 'images')
        save_det_crops(images_dir, gt_dir, det_dir, save_dir)


