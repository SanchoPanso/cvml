import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Object-Detection-Metrics', 'lib'))


from .custom_bounding_boxes import CustomBoundingBoxes
from .custom_evaluator import CustomEvaluator
from .metrics_utils import get_bounding_boxes
from utils import *
from BoundingBox import *
from BoundingBoxes import *
from Evaluator import *


# # redefinition for improved classes
# BoundingBoxes = CustomBoundingBoxes
# Evaluator = CustomEvaluator


