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
    
    def __add__(self, other):
        
        sum_classes = []
        
        if self.classes == other.classes:
            sum_classes = self.classes
        elif len(self.classes) == 0:
            sum_classes = other.classes
        elif len(other.classes) == 0:
            sum_classes = self.classes
        else:
            raise ValueError("Wrong classes") # redo
        
        sum_bbox_map = {} 
        for name in other.bbox_map:
            sum_bbox_map[name] = other.bbox_map[name]
        for name in self.bbox_map:
            sum_bbox_map[name] = self.bbox_map[name]
        
        return Annotation(sum_classes, sum_bbox_map)