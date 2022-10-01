import os
import numpy as np
from abc import ABC


class Segmenter(ABC):
    def __init__(self, weights_path: str, device: str):
        pass

    def __call__(self, img: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    