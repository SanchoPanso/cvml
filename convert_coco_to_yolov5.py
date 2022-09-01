import json
import random
import os
import shutil


class Extractor:
    def __init__(self):
        pass

    def extract(self, input_annotation: str) -> dict:
        with open(input_annotation) as f:
            input_data = json.load(f)

        classes = self.get_classes(input_data)
        images = self.get_images(input_data)
        bboxes = self.get_bboxes(input_data)

        annotations = {}
        for i, bbox_id in enumerate(bboxes.keys()):
            print('{0}/{1}'.format(i + 1, len(bboxes.keys())))

            record = self.get_bbox_record(bbox_id, classes, images, bboxes)
            file_name, cls_num, x, y, w, h = record

            if file_name not in annotations.keys():
                annotations[file_name] = []

            annotations[file_name].append([cls_num, x, y, w, h])

        result = {
            'classes': classes,
            'annotations': annotations,
        }
        return result

    def get_classes(self, input_data: dict) -> dict:
        input_categories = input_data['categories']
        result = {}
        for cls_num in range(len(input_categories)):
            cls_id = input_categories[cls_num]['id']
            cls_name = input_categories[cls_num]['name']
            result[cls_id] = {
                'cls_num': cls_num,
                'cls_name': cls_name,
            }
        return result

    def get_images(self, input_data: dict) -> dict:
        input_images = input_data['images']
        result = {}
        for images_num in range(len(input_images)):
            image_id = input_images[images_num]['id']
            width = input_images[images_num]['width']
            height = input_images[images_num]['height']
            file_name = input_images[images_num]['file_name']
            result[image_id] = {
                'image_id': image_id,
                'width': width,
                'height': height,
                'file_name': file_name,
            }
        return result

    def get_bboxes(self, input_data: dict) -> dict:
        input_annotations = input_data['annotations']
        result = {}
        for bbox_num in range(len(input_annotations)):
            bbox_id = input_annotations[bbox_num]['id']
            image_id = input_annotations[bbox_num]['image_id']
            cls_id = input_annotations[bbox_num]['category_id']
            bbox = input_annotations[bbox_num]['bbox']
            result[bbox_id] = {
                'bbox_num': bbox_num,
                'image_id': image_id,
                'cls_id': cls_id,
                'bbox': bbox,
            }
        return result

    def get_bbox_record(self, bbox_id: str, classes: dict, images: dict, bboxes: dict) -> tuple:
        bbox = bboxes[bbox_id]['bbox']
        image_id = bboxes[bbox_id]['image_id']
        cls_id = bboxes[bbox_id]['cls_id']

        cls_num = classes[cls_id]['cls_num']

        width = images[image_id]['width']
        height = images[image_id]['height']
        file_name = images[image_id]['file_name']

        x, y, w, h = bbox
        x = (x + w/2) / width
        y = (y + h/2) / height
        w /= width
        h /= height

        return file_name, cls_num, x, y, w, h


class Installer:
    def __init__(self):
        self.train_percentage = 0.7
        self.valid_percentage = 0.2

    def install(self, data: dict, images_dir: str, install_dir: str):
        classes = data['classes']
        annotations = data['annotations']

        split_dict = self.get_split_dict(list(annotations.keys()))

        for split_name in ['train', 'valid', 'test']:
            for data_type in ['labels', 'images']:
                os.makedirs(os.path.join(install_dir, split_name, data_type), exist_ok=True)

        for i, img_file_name in enumerate(annotations.keys()):
            print('{0}/{1}'.format(i + 1, len(annotations.keys())))

            name = ''.join(img_file_name.split('.')[:-1])
            split_name = split_dict[img_file_name]

            shutil.copy(os.path.join(images_dir, img_file_name),
                        os.path.join(install_dir, split_name, 'images', img_file_name))

            with open(os.path.join(install_dir, split_name, 'labels', f'{name}.txt'), 'w') as f:
                lines = annotations[img_file_name]
                for line in lines:
                    f.write(' '.join(list(map(str, line))) + '\n')

        with open(os.path.join(install_dir, 'data.yaml'), 'w') as f:
            data_yaml_text = f"train: ../train/images\n" \
                             f"val: ../valid/images\n\n" \
                             f"nc: {len(classes.keys())}\n" \
                             f"names: {[classes[key]['cls_name'] for key in classes.keys()]}\n"
            f.write(data_yaml_text)

    def get_split_dict(self, img_file_names: list) -> dict:
        img_file_names_copy = img_file_names.copy()
        random.shuffle(img_file_names_copy)

        result = {}

        for i, img_file_name in enumerate(img_file_names_copy):
            if 0 <= i / len(img_file_names_copy) < self.train_percentage:
                result[img_file_name] = 'train'
            elif self.train_percentage <= i / len(img_file_names_copy) < self.train_percentage + self.valid_percentage:
                result[img_file_name] = 'valid'
            else:
                result[img_file_name] = 'test'

        return result


if __name__ == '__main__':
    images_dir = r'C:\Users\Alex\Downloads\Номера+дефекты\Номера+дефекты\defects1\defects1'
    coco_path = r'C:\Users\Alex\Downloads\defects1.json'
    install_dir = r'C:\Users\Alex\Downloads\defects1_test'

    extractor = Extractor()
    installer = Installer()

    data = extractor.extract(coco_path)
    print(data['annotations'])
    installer.install(data, images_dir, install_dir)

