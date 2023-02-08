import os
import numpy as np
from abc import ABC
from typing import List
from cvml.annotation.annotation import BoundingBox


class Detector(ABC):
    def __init__(self, weights_path: str, device: str):
        pass

    def __call__(self, img: np.ndarray, *args, **kwargs) -> List[BoundingBox]:
        pass
    


