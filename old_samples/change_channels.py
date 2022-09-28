import os
import cv2
from pip import main

def change_channels(img):
    channels = list(cv2.split(img))
    channels[0], channels[1] = channels[1], channels[0]
    new_img = cv2.merge(channels)
    return new_img


if __name__ == '__main__':
    dataset_dir = 'E:\PythonProjects\AnnotationConverter\datasets\segment_26082022'
    new_dataset_dir = 'E:\PythonProjects\AnnotationConverter\datasets\segment_26082022_rev'
    for split in ['train', 'valid']:
        os.makedirs(os.path.join(new_dataset_dir, split), exist_ok=True)
        dir_path = os.path.join(dataset_dir, split)
        files = os.listdir(dir_path)
        for file in files:
            print(file)
            img = cv2.imread(os.path.join(dir_path, file))
            if img is None:
                continue
            new_img = change_channels(img)
            cv2.imwrite(os.path.join(new_dataset_dir, split, file), new_img)
        
