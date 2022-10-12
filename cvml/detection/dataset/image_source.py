import os
import cv2
from abc import ABC
from typing import List, Callable
import numpy as np


class ImageReader(ABC):
    def set_params(self, *args, **kwargs):
        pass

    def get_name(self) -> str:
        pass

    def read(self) -> np.ndarray:
        pass


class SingleImageReader(ImageReader):
    def __init__(self):
        self.path = None
        self.preprocess_fn = None
    
    def get_name(self):
        filename = os.path.split(self.path)[-1]
        name, ext = os.path.splitext(filename)
        return name

    def set_params(self, path: str, preprocess_fn: Callable = None):
        self.path = path
        self.preprocess_fn = preprocess_fn
    
    def read(self) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = self.preprocess_fn(img)
        return img


class MultipleImageReader(ImageReader):
    def __init__(self):
        self.paths = None
        self.main_channel = None
        self.preprocess_fns = None
    
    def get_name(self):
        main_path = self.paths[self.main_channel]
        filename = os.path.split(main_path)[-1]
        name, ext = os.path.splitext(filename)
        return name

    def set_params(self, paths: List[str], main_channel: int = 0, preprocess_fns: List[Callable] = None):
        self.paths = paths
        self.main_channel = main_channel
        self.preprocess_fns = [lambda x:x] * len(paths) if preprocess_fns is None else preprocess_fns

    def read(self):
        imgs = []
        for i in range(len(self.paths)):
            img = cv2.imdecode(np.fromfile(self.paths[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if self.preprocess_fns[i] is not None:
                img = self.preprocess_fns[i](img)
            imgs.append(img)

        final_img = cv2.merge(imgs)
        return final_img


class ImageSource(ABC):
    def get_name(self) -> str:
        pass

    def save(self, path: str):
        pass


class DetectionImageSource(ImageSource):
    def __init__(self, image_reader: ImageReader, *args, **kwargs):
        self.image_reader = image_reader
        self.image_reader.set_params(*args, **kwargs)

    def get_name(self):
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


def convert_paths_to_sources(paths: List[List[str]], preprocess_fns: List[Callable], main_channel: int):

    if len(paths) == 0 or len(paths[0]) == 0:
        return []
    
    for i in range(1, len(paths)):
        if len(paths[i - 1]) != len(paths[i]):
            raise ValueError("Number of each channels paths must be the same")

    image_sources = []
    num_of_channels = len(paths)
    num_of_sources = len(paths[0])

    for i in range(num_of_sources):
        cur_paths = [paths[channel][i] for channel in range(num_of_channels)]
        image_source = DetectionImageSource(MultipleImageReader(), cur_paths, main_channel, preprocess_fns)
        image_sources.append(image_source)

    return image_sources
