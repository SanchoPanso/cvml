import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Object-Detection-Metrics', 'samples', 'sample_2'))

import _init_paths
from BoundingBoxes import BoundingBoxes


class CustomBoundingBoxes(BoundingBoxes):
    def __init__(self, bounding_boxes: list = None):
        if bounding_boxes is None:
            self._boundingBoxes = []
        else:
            self._boundingBoxes = bounding_boxes

    def __getitem__(self, key):
        return self._boundingBoxes[key]

    def __len__(self):
        return self._boundingBoxes.__len__()

    def __add__(self, other):
        return CustomBoundingBoxes(self._boundingBoxes + other.getBoundingBoxes())



