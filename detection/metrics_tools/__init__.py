import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Object-Detection-Metrics', 'samples', 'sample_2'))

import _init_paths
from BoundingBox import BoundingBox
from .custom_bounding_boxes import CustomBoundingBoxes
from .custom_evaluator import CustomEvaluator
from utils import *


# # redefinition for improved classes
# BoundingBoxes = CustomBoundingBoxes
# Evaluator = CustomEvaluator


