import os
import cv2
import numpy as np
from installer import Installer
import random
import shutil

raw_dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_segment_15082022_pol_raw'
dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_segment_15082022'

files = list(os.listdir(raw_dataset_dir))
names = set()
for file in files:
    name = file.split('.')[0]
    name = name.replace('_mask', '')
    names.add(name)

names = list(names)
random.shuffle(names)

lenght = len(names)
print(lenght)
files_dict = {
    'train': set(names[0:int(lenght * 0.8)]),
    'valid': set(names[int(lenght * 0.8):]),
}


for split in ['train', 'valid']:
    os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
    for i, name in enumerate(files_dict[split]):
        shutil.copy(os.path.join(raw_dataset_dir, name + '.jpg'),
                    os.path.join(dataset_dir, split, name + '.jpg'))

        mask = cv2.imread(os.path.join(raw_dataset_dir, name + '_mask.jpg'))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
        np.savez(os.path.join(dataset_dir, split, name + '.npz'), arr_0=mask)
        print(i, name)



