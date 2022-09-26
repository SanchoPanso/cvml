import os
import cv2
from abc import ABC
from typing import List, Callable
import numpy as np


class ImageSource(ABC):
    def get_name(self) -> str:
        pass

    def read(self) -> np.ndarray:
        pass


class SingleImageSource(ImageSource):
    def __init__(self, path: str, preprocess_fn: Callable = None):
        self.path = path
        self.preprocess_fn = preprocess_fn

    def get_name(self):
        filename = os.path.split(self.path)[-1]
        name, ext = os.path.splitext(filename)
        return name

    def read(self):
        img = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return img


class MultipleImageSource(ImageSource):
    def __init__(self, paths: List[str], main_channel: int = 0, preprocess_fns: List[Callable] = None):
        self.paths = paths
        self.main_channel = main_channel
        self.preprocess_fns = [lambda x:x] * len(paths) if preprocess_fns is None else preprocess_fns

    def get_name(self):
        main_path = self.paths[self.main_channel]
        filename = os.path.split(main_path)[-1]
        name, ext = os.path.splitext(filename)
        return name

    def read(self):
        imgs = []
        for i in range(len(self.paths)):
            img = cv2.imdecode(np.fromfile(self.paths[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if self.preprocess_fns[i] is not None:
                img = self.preprocess_fns[i](img)
            imgs.append(img)

        final_img = cv2.merge(imgs)
        return final_img


def convert_paths_to_sources(paths: List[List[str]], preprocess_fns: List[Callable], main_channel: int):

    if len(paths) == 0 or len(paths[0]) == 0:
        return []

    image_sources = []
    num_of_channels = len(paths)
    num_of_sources = len(paths[0])

    for i in range(num_of_sources):
        cur_paths = [paths[channel][i] for channel in range(num_of_channels)]
        image_source = MultipleImageSource(cur_paths, main_channel, preprocess_fns)
        image_sources.append(image_source)

    return image_sources
