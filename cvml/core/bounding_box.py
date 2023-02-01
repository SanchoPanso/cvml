from enum import Enum
from typing import Tuple


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
    """
    Relative = 1
    Absolute = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GroundTruth = 1
    Detected = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2
    """
    XYWH = 1
    XYX2Y2 = 2


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convertToRelativeValues(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convertToAbsoluteValues(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


class BoundingBox:
    def __init__(self,
                 class_id: int = 0,
                 x: float = 0.0,
                 y: float = 0.0,
                 w: float = 0.0,
                 h: float = 0.0,
                 class_confidence: float = None,
                 image_name: str = None,
                 type_coordinates: CoordinatesType = CoordinatesType.Absolute,
                 img_size: tuple = None,
                 bb_type: BBType = BBType.GroundTruth,
                 format: BBFormat = BBFormat.XYWH,
                 segmentation: dict or list = None):
        """Constructor.
        Args:
            image_name: String representing the image name.
            class_id: String value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box
            coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the
            image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box
            represents a ground truth or a detection. If it is a detection, the classConfidence has
            to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected
            class. If detectionType is Detection, classConfidence needs to be informed.
            format: (optional) Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the
            coordinates of the bounding boxes. BBFormat.XYWH: <left> <top> <width> <height>
            BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        
        if bb_type == BBType.Detected and class_confidence is None:
            raise ValueError(
                'For bbType=\'Detection\', it is necessary to inform the classConfidence value.')
        
        if class_confidence != None and (class_confidence < 0 or class_confidence > 1):
            raise IOError('classConfidence value must be a real value between 0 and 1. Value: %f' %
            class_confidence)

        self._image_name = image_name
        self._class_confidence = class_confidence
        self._bb_type = bb_type
        self._class_id = class_id
        
        self.set_coordinates((x, y, w, h), format, type_coordinates, img_size)
        self._segmentation = segmentation or []

    def get_coordinates(self,
                        coordinates_type: CoordinatesType = CoordinatesType.Absolute, 
                        format: BBFormat = BBFormat.XYWH,
                        img_size: tuple = None):
        
        if coordinates_type == CoordinatesType.Absolute:
            return self.get_absolute_bounding_box(format)
        return self.get_relative_bounding_box(img_size)
    
    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None):
        if img_size is None and self._width_img is None and self._height_img is None:
            raise ValueError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if img_size is not None:
            return convertToRelativeValues((img_size[0], img_size[1]),
                                           (self._x, self._x2, self._y, self._y2))
        else:
            return convertToRelativeValues((self._width_img, self._height_img),
                                           (self._x, self._x2, self._y, self._y2))

    def get_image_name(self) -> str:
        return self._image_name

    def get_confidence(self) -> float:
        return self._class_confidence

    def get_class_id(self) -> int:
        return self._class_id

    def get_image_size(self) -> tuple:
        return self._width_img, self._height_img

    def get_bb_type(self) -> BBType:
        return self._bb_type
    
    def get_segmentation(self):
        return self._segmentation
    
    def set_coordinates(self,
                        coordinates: Tuple[float],
                        format=BBFormat.XYWH,
                        coordinates_type=CoordinatesType.Absolute,
                        img_size: tuple = None):
        
        if coordinates_type == CoordinatesType.Relative and img_size is None:
            raise ValueError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        
        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (coordinates_type == CoordinatesType.Relative):
            x, y, w, h = coordinates
            (self._x, self._y, self._w, self._h) = convertToAbsoluteValues(img_size, (x, y, w, h))
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if format == BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise ValueError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        
        # For absolute coords: (x,y,w,h) = real bb coords
        else:
            if format == BBFormat.XYWH:
                x, y, w, h = coordinates
                self._x = x
                self._y = y
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                x1, y1, x2, y2 = coordinates
                self._x = x
                self._y = y
                self._x2 = x2
                self._y2 = y2
                self._w = x2 - x1
                self._h = y2 - y1
                
        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

    def set_image_name(self, image_name: str):
        self._image_name = image_name

    def set_confidence(self, confidence: float):
        self._class_confidence = confidence

    def set_class_id(self, class_id: int):
        self._class_id = class_id

    def set_image_size(self, image_size: tuple):
        self._width_img, self._height_img = image_size

    def set_bb_type(self, bb_type: BBType):
        self._bb_type = bb_type
    
    def set_segmentation(self, segmentation):
        self._segmentation = segmentation
    
    def __str__(self):
        return f"BoundingBox(class_id = {self._class_id}, x = {self._x}, y = {self._y}, w = {self._w}, h = {self._h})"

    def __eq__(self, other):
        det1BB = self.getAbsoluteBoundingBox()
        det1ImgSize = self.getImageSize()
        det2BB = other.getAbsoluteBoundingBox()
        det2ImgSize = other.getImageSize()

        if self.getClassId() == other.getClassId() and \
           self.classConfidence == other.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1ImgSize[0] == det2ImgSize[0] and \
           det1ImgSize[1] == det2ImgSize[1]:
            return True
        return False

    def clone(self):
        absBB = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        newBoundingBox = BoundingBox(self.get_image_name(),
                                     self.get_class_id(),
                                     absBB[0],
                                     absBB[1],
                                     absBB[2],
                                     absBB[3],
                                     type_coordinates=CoordinatesType.Absolute,
                                     img_size=self.get_image_size(),
                                     bb_type=self.get_bb_type(),
                                     class_confidence=self.get_confidence(),
                                     format=BBFormat.XYWH)
        return newBoundingBox


