import os

from detection.inference_tools.yolov5_detector import Yolov5Detector
from detection.inference_tools.visualizer import Visualiser

model_path = r'C:\Users\Alex\Downloads\yolov5l_250ep_17092022-20220920T205028Z-001\yolov5l_250ep_17092022\weights\best.pt'
dataset_dir = r'F:\datasets\tmk_yolov5_17092022'
save_dir = r'F:\datasets\tmk_yolov5_17092022_visalization'

detector = Yolov5Detector(model_path)

visualizer = Visualiser()
visualizer.splits = ['train']
visualizer.classes = {0: 'comet', 2: 'joint', 3: 'number'}
visualizer.pred_setting = {'conf': 0.3, 'count_part_x': 1, 'count_part_y': 1}

visualizer.create_visualization(dataset_dir, save_dir, detector)

