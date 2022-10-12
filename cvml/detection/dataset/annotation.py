from typing import List, Dict
from cvml.core.bounding_box import BoundingBox


class Annotation:
    """
    Representation of the detection annotation of some dataset.
    """
    def __init__(self, 
                 classes: List[str] = None, 
                 bbox_map: Dict[str, List[BoundingBox]] = None):
        """
        :classes: list of class names
        :bbox_map: dict with keys - image names and values - list of bounding boxes on this image
        """
        
        self.classes = [] if classes is None else classes
        self.bbox_map = {} if bbox_map is None else bbox_map