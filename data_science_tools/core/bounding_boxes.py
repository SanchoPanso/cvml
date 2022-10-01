from .bounding_box import BoundingBox


class BoundingBoxes:
    def __init__(self, bounding_boxes: list = None):
        if bounding_boxes is None:
            self._boundingBoxes = []
        else:
            self._boundingBoxes = bounding_boxes

    def __str__(self):
        string = '['
        for i, bb in enumerate(self._boundingBoxes):
            string += str(bb)
            if i != len(self._boundingBoxes) - 1:
                string += ', '
        string += ']'
        return string

    def __getitem__(self, key):
        return self._boundingBoxes[key]

    def __len__(self):
        return self._boundingBoxes.__len__()

    def __add__(self, other):
        return BoundingBoxes(self._boundingBoxes + other.getBoundingBoxes())

    def append(self, bb):
        self._boundingBoxes.append(bb)

    def remove(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d, _boundingBox):
                del self._boundingBoxes[d]
                return

    def remove_all(self):
        self._boundingBoxes = []

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName]

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    # def drawAllBoundingBoxes(self, image, imageName):
    #     bbxes = self.getBoundingBoxesByImageName(imageName)
    #     for bb in bbxes:
    #         if bb.getBBType() == BBType.GroundTruth:  # if ground truth
    #             image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
    #         else:  # if detection
    #             image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
    #     return image

    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image
