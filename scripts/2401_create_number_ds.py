import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cvml.detection.dataset.annotation_converter import AnnotationConverter, Annotation


def main():
    conformity = {
        '23_06_2021_номера_оправок_командир': 'number_0',
        'cvs1_number1': 'number_1',
        'cvs1_number2': 'number_2',
        'cvs1_number3': 'number_3',
    }
    number_crop_dir = '/home/student2/Downloads/gray_class'
    datasets_dir = '/home/student2/datasets/raw'

    match_cnt = 0
    
    for dataset_name in conformity.keys():
        dataset_dir = os.path.join(datasets_dir, dataset_name)
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        annotation = AnnotationConverter.read_coco(annotation_path)
        new_annotation = Annotation([str(i) for i in range(9)], {})
        
        for image_name in annotation.bbox_map:
            new_annotation.bbox_map[image_name] = []
            bboxes = annotation.bbox_map[image_name]
            crop_cnt = 1
            
            for bbox in bboxes:
                if bbox.get_class_id() != 3:    #CHECK
                    continue
                old_dataset_name = conformity[dataset_name]
                crop_name = f'{image_name}_{old_dataset_name}_{crop_cnt}.png'
                
                for digit in range(9):
                    if os.path.exists(os.path.join(number_crop_dir, str(digit), crop_name)):
                        bbox._class_id = digit
                        new_annotation.bbox_map[image_name].append(bbox)
                        match_cnt += 1
                        print(crop_name)
                        break
                crop_cnt += 1
        
        AnnotationConverter.write_coco(new_annotation, os.path.join(dataset_dir, 'annotations', 'digits.json'), '.png')
        print(match_cnt)

                
if __name__ == '__main__':
    main()