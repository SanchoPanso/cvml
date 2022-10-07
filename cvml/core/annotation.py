from typing import List, Dict
from cvml.core.bounding_box import BoundingBox


class Annotation:
    """
    Representation of the detection annotation of some dataset.
    """
    def __init__(self, 
                 classes: List[str] = None, 
                 bounding_boxes: Dict[str, List[BoundingBox]] = None):
        """
        :classes: list of class names
        :bounding_boxes: dict with keys - image names and values - list of bounding boxes on this image
        """
        
        self.classes = [] if classes is None else classes
        self.bounding_boxes = {} if bounding_boxes is None else bounding_boxes