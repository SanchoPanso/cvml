import os
import cv2
from abc import ABC
from typing import List, Callable
import numpy as np


class ImageReader(ABC):
    """Implementor of image reading functionality.
    """
    
    def get_name(self, paths: List[str]) -> str:
        """Return name for following save name based on original paths

        Args:
            paths (List[str]): source paths of images

        Returns:
            str: name
        """
        pass

    def read(self, paths: List[str], preprocessing_fns: List[Callable]) -> np.ndarray:
        """Read and preprocess image specifically 

        Args:
            paths (List[str]): source paths of images
            preprocessing_fns (List[Callable]): list of preprocessing functions

        Returns:
            np.ndarray: read and preprocessed image
        """
        pass


class SingleImageReader(ImageReader):
    """Concrete implementor for reading a single image for a single image source
    """
    def __init__(self):
        pass
    
    def get_name(self, paths: List[str]) -> str:
        filename = os.path.split(paths[0])[-1]
        name, ext = os.path.splitext(filename)
        return name
    
    def read(self, paths: List[str],  preprocessing_fns: List[Callable]) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        img = preprocessing_fns[0](img)
        return img


class MultipleImageReader(ImageReader):
    """Concrete implementor for reading some images for a single image source
    """
    def __init__(self, main_channel: int = 0):
        self.main_channel = main_channel
        
    def get_name(self, paths: List[str]) -> str:
        main_path = paths[self.main_channel]
        filename = os.path.split(main_path)[-1]
        name, ext = os.path.splitext(filename)
        return name

    def read(self, paths: List[str], preprocessing_fns: List[Callable]) -> np.ndarray:
        imgs = []
        for i in range(len(paths)):
            img = cv2.imdecode(np.fromfile(paths[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if preprocessing_fns[i] is not None:
                img = preprocessing_fns[i](img)
            imgs.append(img)

        final_img = cv2.merge(imgs)
        return final_img


class ImageSource(ABC):
    """Abstraction of image source
    """
    name = None
    
    def __init__(self, paths: List[str] = None, preprocessing_fns: List[Callable] = None, image_reader: ImageReader = None, name: str = None):
        pass
    
    def read(self) -> np.ndarray:
        pass
    
    def save(self, path: str):
        pass


class DetectionImageSource(ImageSource):
    """Refined abstraction of ImageSource for detection dataset
    """
    def __init__(self, paths: List[str] = None, preprocessing_fns: List[Callable] = None, image_reader: ImageReader = None, name: str = None):
        
        if len(paths) == 0:
            raise ValueError("List path is empty.")
        if len(paths) != len(preprocessing_fns):
            raise ValueError("Number of paths is not equal to number of preprocessing_fns")
         
        self.paths = paths
        self.preprocessing_fns = preprocessing_fns
        self.image_reader = image_reader
        self.name = name or self.image_reader.get_name(self.paths) 

    # def get_name(self):
    #     return self.image_reader.get_name(self.paths)

    def read(self) -> np.ndarray:
        return self.image_reader.read(self.paths, self.preprocessing_fns)
    
    def _write(self, path: str, img: np.ndarray):
        ext = os.path.splitext(os.path.split(path)[-1])[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        im_buf_arr.tofile(path)
    
    def save(self, path: str):
        img = self.read()
        self._write(path, img)


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



def convert_paths_to_multiple_sources(paths: List[List[str]], 
                                      preprocess_fns: List[Callable], 
                                      main_channel: int) -> List[ImageSource]:

    if len(paths) == 0 or len(paths[0]) == 0:
        return []
    
    for i in range(1, len(paths)):
        if len(paths[i - 1]) != len(paths[i]):
            raise ValueError("Number of each channels paths must be the same")

    image_sources = []
    num_of_channels = len(paths)
    num_of_sources = len(paths[0])

    image_reader = MultipleImageReader(main_channel)
    for i in range(num_of_sources):
        cur_paths = [paths[channel][i] for channel in range(num_of_channels)]
        image_source = DetectionImageSource(cur_paths, preprocess_fns, image_reader)
        image_sources.append(image_source)

    return image_sources


def convert_paths_to_single_sources(paths: List[str], 
                                    preprocess_fn: Callable) -> List[ImageSource]:
    
    image_sources = []
    image_reader = SingleImageReader()
    for i, path in enumerate(paths):
        image_source = DetectionImageSource([path], [preprocess_fn], image_reader)
        image_sources.append(image_source)

    return image_sources


# old
def convert_single_paths_to_sources(paths: List[str], preprocess_fn: Callable):
    
    image_sources = []

    for i, path in enumerate(paths):
        image_source = DetectionImageSource(SingleImageReader(), path, preprocess_fn)
        image_sources.append(image_source)

    return image_sources
