import os
import shutil
import random


class Installer:
    def __init__(self):
        self.train_percentage = 0.7
        self.valid_percentage = 0.2

    def install(self, data: dict, images_dir: str, install_dir: str):
        classes = data['classes']
        annotations = data['annotations']

        # assign split name to every image
        split_dict = self.get_split_dict(list(annotations.keys()))

        for split_name in ['train', 'valid', 'test']:
            for data_type in ['labels', 'images']:
                os.makedirs(os.path.join(install_dir, split_name, data_type), exist_ok=True)

        for i, img_file_name in enumerate(annotations.keys()):
            print('{0}/{1}'.format(i + 1, len(annotations.keys())))

            name = ''.join(img_file_name.split('.')[:-1])
            split_name = split_dict[img_file_name]

            # copy image to install dir
            shutil.copy(os.path.join(images_dir, img_file_name),
                        os.path.join(install_dir, split_name, 'images', img_file_name))

            # write txt-file with labels
            self.write_label_file(os.path.join(install_dir, split_name, 'labels'),
                                  f'{name}.txt',
                                  annotations[img_file_name])

        self.write_description(classes, install_dir)

        # with open(os.path.join(install_dir, 'data.yaml'), 'w') as f:
        #     data_yaml_text = f"train: ../train/images\n" \
        #                      f"val: ../valid/images\n\n" \
        #                      f"nc: {len(classes.keys())}\n" \
        #                      f"names: {[classes[key]['cls_name'] for key in classes.keys()]}\n"
        #     f.write(data_yaml_text)

    def install_with_anchor(self, data: dict, images_dir: str, install_dir: str, anchor_dataset_dir: str):
        classes = data['classes']
        annotations = data['annotations']

        splits = ['test', 'train', 'valid']
        orig_images = os.listdir(images_dir)
        for split in splits:
            anchor_images = os.listdir(os.path.join(anchor_dataset_dir, split, 'images'))
            os.makedirs(os.path.join(install_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(install_dir, split, 'labels'), exist_ok=True)
            for image in anchor_images:
                print(split, image)
                shutil.copy(os.path.join(images_dir,
                                         os.path.splitext(image)[0] + '.jpg'),
                            os.path.join(install_dir, split, 'images',
                                         os.path.splitext(image)[0] + '.jpg'))
                self.write_label_file(os.path.join(install_dir, split, 'labels'),
                                      os.path.splitext(image)[0] + '.txt',
                                      data['annotations'][image])

    def write_label_file(self, dir: str, txt_file_name: str, lines: list):
        with open(os.path.join(dir, txt_file_name), 'w') as f:
            for line in lines:
                f.write(' '.join(list(map(str, line))) + '\n')

    def copy_image_file(self, src_path: str, dst_path: str):
        shutil.copy(src_path, dst_path)

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

    def write_description(self, classes: list, dir: str):
        text = f"train: ../train/images\n" \
               f"val: ../valid/images\n\n" \
               f"nc: {len(classes)}\n" \
               f"names: {classes}"
        with open(os.path.join(dir, 'data.yaml'), 'w') as f:
            f.write(text)


