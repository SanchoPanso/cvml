import os
import cv2
from abc import ABC
from typing import List, Callable
import numpy as np

from cvml.detection.dataset.image_source import ImageReader, ImageSource


class ISImageSource(ImageSource):
    def __init__(self, image_reader: ImageReader, color_mask_path: str, *args, **kwargs):
        self.image_reader = image_reader
        self.image_reader.set_params(*args, **kwargs)
        self.color_mask_path = color_mask_path

    def get_color_mask(self) -> np.ndarray:
        if self.color_mask_path == None:
            return None

        img = cv2.imread(self.color_mask_path)
        return img

    def get_name(self) -> str:
        return self.image_reader.get_name()

    def read(self) -> np.ndarray:
        self.image_reader.read()
    
    def write(self, path: str, img: np.ndarray):
        ext = os.path.splitext(os.path.split(path)[-1])[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        im_buf_arr.tofile(path)
    
    def save(self, path: str):
        img = self.read()
        self.write(path, img)


# def convert_paths_to_sources(paths: List[List[str]], preprocess_fns: List[Callable], main_channel: int):

#     if len(paths) == 0 or len(paths[0]) == 0:
#         return []
    
#     for i in range(1, len(paths)):
#         if len(paths[i - 1]) != len(paths[i]):
#             raise ValueError("Number of each channels paths must be the same")

#     image_sources = []
#     num_of_channels = len(paths)
#     num_of_sources = len(paths[0])

#     for i in range(num_of_sources):
#         cur_paths = [paths[channel][i] for channel in range(num_of_channels)]
#         image_source = DetectionImageSource(MultipleImageReader(), cur_paths, main_channel, preprocess_fns)
#         image_sources.append(image_source)

#     return image_sources
