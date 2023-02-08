import os
import sys
from glob import glob
import cv2
import numpy as np
from typing import Callable, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from cvml.dataset.image_source import MultipleImageReader
from cvml.instance_segmentation.dataset.instance_segmentation_dataset import ISDataset, ISImageSource
from cvml.detection.dataset.annotation_converter import AnnotationConverter
from cvml.detection.dataset.annotation_editor import AnnotationEditor

from cvml.dataset.image_transforming import expo

source_dir = '/home/student2/datasets/raw/TMK_CVS3'

comet_1_dir = os.path.join(source_dir, 'csv1_comets_1_24_08_2022')
comet_2_dir = os.path.join(source_dir, 'csv1_comets_2_24_08_2022')
comet_3_dir = os.path.join(source_dir, 'csv1_comets_23_08_2022')
comet_4_dir = os.path.join(source_dir, 'csv1_comets_01_09_2022')
comet_5_dir = os.path.join(source_dir, 'csv1_comets_05_09_2022')


dataset_dirs = {
    'tmk_1': os.path.join(source_dir, 'SVC3_defects TMK1'),
    'tmk_2': os.path.join(source_dir, 'SVC3_defects TMK2'),
    'tmk_3': os.path.join(source_dir, 'SVC3_defects TMK3'),
    'tmk_4': os.path.join(source_dir, 'SVC3_defects TMK4'),
    'tmk_5': os.path.join(source_dir, 'SVC3_defects TMK5'),
}

renamers = {}
for key in dataset_dirs.keys():
    renamers[key] = lambda x: x + '_' + key


def wrap_expo(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = expo(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rename_annotation_files(annotations_data: dict,  rename_callback: Callable) -> dict:
    names = list(annotations_data['annotations'].keys())
    new_annotations_data = {'classes': annotations_data['classes'], 'annotations': {}}
    for name in names:
        new_name = rename_callback(name)

        labels = annotations_data['annotations'][name]
        new_annotations_data['annotations'][new_name] = labels

    return new_annotations_data


def get_image_number(img_path: str) -> int:
    img_filename = os.path.split(img_path)[-1]
    name, ext = os.path.splitext(img_filename)
    num_str = name.split('_')[0]
    num = int(num_str)
    return num


def convert_paths_to_is_sources(paths: List[List[str]], 
                                color_mask_paths: List[str], 
                                preprocess_fns: List[Callable], 
                                main_channel: int):

    if len(paths) == 0 or len(paths[0]) == 0:
        return []
    
    for i in range(1, len(paths)):
        if len(paths[i - 1]) != len(paths[i]):
            raise ValueError("Number of each channels paths must be the same")

    image_sources = []
    num_of_channels = len(paths)
    num_of_sources = len(paths[0])

    for i in range(num_of_sources):
        cur_paths = [paths[channel][i] for channel in range(num_of_channels)]
        color_mask_path = color_mask_paths[i]
        image_source = ISImageSource(MultipleImageReader(), color_mask_path, cur_paths, main_channel, preprocess_fns)
        image_sources.append(image_source)

    return image_sources


if __name__ == '__main__':

    converter = AnnotationConverter()
    editor = AnnotationEditor()
    final_dataset = ISDataset()

    for key in dataset_dirs.keys():
        
        dataset = ISDataset()

        dataset_dir = dataset_dirs[key]
        print(dataset_dir)

        image_dir = os.path.join(dataset_dir, 'images')
        polarization_dir = os.path.join(dataset_dir, 'polar')

        color_mask_paths = glob(os.path.join(image_dir, '*color*'))
        channel_0_paths = glob(os.path.join(polarization_dir, '*_1.*'))
        channel_1_paths = glob(os.path.join(polarization_dir, '*_2.*'))
        channel_2_paths = list(set(glob(os.path.join(image_dir, '*'))) - set(color_mask_paths))

        color_mask_paths.sort(key=lambda x: get_image_number(x))
        channel_0_paths.sort(key=lambda x: get_image_number(x))
        channel_1_paths.sort(key=lambda x: get_image_number(x))
        channel_2_paths.sort(key=lambda x: get_image_number(x))

        expanded_color_mask_paths = []
        mask_numbers = set([get_image_number(p) for p in color_mask_paths])
        for img_file in channel_2_paths:
            num = str(get_image_number(img_file))
            found_masks = glob(os.path.join(image_dir, f'{num}_color*'))
            if len(found_masks) == 0:
                expanded_color_mask_paths.append(None)
            else:
                expanded_color_mask_paths.append(found_masks[0])

        image_sources = convert_paths_to_is_sources([channel_0_paths, channel_1_paths, channel_2_paths],
                                                    expanded_color_mask_paths,
                                                    main_channel=2,
                                                    preprocess_fns=[None, None, wrap_expo])

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = renamers[key]

        annotation_data = converter.read_coco(annotation_path)

        changes = {
            0: None,    # comet 
            1: None,       # other
            2: None,    # joint 
            3: None,    # number
            4: None,       # tube
            #5: 5,       # sink
            6: None,    # birdhouse
            7: None,    # print
            #8: 8,       # riska
            #9: 9,       # deformation defect
            #10: 10,     # continuity violation
        }
        annotation_data = editor.change_classes_by_id(annotation_data, changes)

        dataset.update(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    final_dataset.classes = ['comet', 'other', 'joint', 'number', 'tube', 'sink', 'birdhouse', 'print', 'riska', 'deformation defect', 'continuity violation']
    result_dir = '/home/student2/datasets/tmk_cvs3_yolov5_07102022'
    # final_dataset.split_by_proportions({'train': 0.7, 'valid': 0.2, 'test': 0.1})
    final_dataset.split_by_dataset('/home/student2/datasets/prepared/tmk_cvs3_yolov5_02102022')
    final_dataset.install(result_dir, install_images=False)


        # for idx in final_dataset.splits['valid']:
    #     lbl_img = final_dataset[idx]
    #     bboxes = lbl_img.bboxes
    #     name = lbl_img.name
    #     for bb in bboxes:
    #         if bb.get_class_id() == 10:
    #             print(name)


