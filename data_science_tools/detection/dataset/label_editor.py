import os
from typing import List


class LabelEditor:
    def __init__(self):
        pass

    def change_classes_in_lines(self, lines: List[List[float or int]], changes: dict) -> list:
        """
        Change class numbers in list of lines of yolov5's annotation
        :param lines: list of lines of yolov5's annotation
        :param changes: dictionary, in which keys - original class number, and values - result class number
        :return: new lines
        """
        new_lines = []
        for line in lines:
            new_line = line
            if changes[int(line[0])] is not None:
                new_line[0] = changes[int(line[0])]
                new_lines.append(new_line)
        return new_lines

    def change_classes(self, data: dict, changes: dict, new_classes: List[str]) -> dict:
        """
        Change classes in annotation dictionary
        :param data: original annotation dictionary
        :param changes: dictionary, in which keys - original class number, and values - result class number
        :param new_classes: list of new class names
        :return: result annotation dictionary
        """
        data['classes'] = new_classes
        for key in data['annotations'].keys():
            lines = data['annotations'][key]
            data['annotations'][key] = self.change_classes_in_lines(lines, changes)
        return data

    def change_image_name(self, data: dict, update_func) -> dict:
        images = list(data['annotations'].keys())
        for image in images:
            name, ext = os.path.splitext(image)
            new_image = update_func(name) + ext
            data['annotations'][new_image] = data['annotations'].pop(image)
        return data
