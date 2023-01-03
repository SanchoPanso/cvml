import os
import sys
import glob
import logging

sys.path.append(os.path.dirname(__file__) + '/..')

from cvml.detection.tools.detection_dataset_creation import create_detection_dataset

raw_datasets_dir = 'D:\\Работа\\СКЗ\\datasets'
raw_dirs = glob.glob(os.path.join(raw_datasets_dir, '*SCV3*'))
raw_dirs.sort()

result_dir = 'D:\\Работа\\СКЗ\\datasets\\test_cvs3'

cls_names = ['other', 'tube', 'sink', 'riska']
split_proportions = {'train': 0.8, 'valid': 0.2, 'test': 0.0}

create_detection_dataset(raw_dirs, result_dir, cls_names, split_proportions)

