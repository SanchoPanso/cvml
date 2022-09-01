import os
from installer import Installer
from extractor import Extractor
from editor import Editor


def change_classes_in_lines(lines: list, change_dict: dict) -> list:
    new_lines = []
    for line in lines:
        new_line = line
        if change_dict[int(line[0])] is not None:
            new_line[0] = change_dict[int(line[0])]
            new_lines.append(new_line)
    return new_lines


images_dir = r'E:\PythonProjects\AnnotationConverter\datasets\numbers2_merged'
annotation_path = r'C:\Users\Alex\Downloads\23_06_2021_numbers_annotations_segment\23_06_2021_numbers_annotations\annotations\instances_default.json'
dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets\numbers2_23082022'

ext = Extractor()
inst = Installer()
edt = Editor()

data = ext.extract(annotation_path)
changes = {
    0: 0,
    1: 1,
    3: 2,
}

data = edt.change_classes(data, changes, ['comet', 'joint', 'number'])

keys = list(data['annotations'].keys())
for key in keys:
    lines = data['annotations'].pop(key)
    new_key = key.split('.')[0] + '.jpg'
    data['annotations'][new_key] = lines

inst.install(data, images_dir, dataset_dir)
