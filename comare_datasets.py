import os
from typing import Set, List
import shutil
from extractor import Extractor
from installer import Installer


def get_file_names(dirs: List[str]) -> Set[str]:
    file_list = []
    for dir in dirs:
        file_list += os.listdir(dir)
    file_set = set(file_list)
    return file_set


if __name__ == '__main__':
    # Define paths to datasets
    datasets_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets')
    defects1_dir = os.path.join(datasets_dir, 'defects1')
    defects2_dir = os.path.join(datasets_dir, 'defects2')
    defects_dir = os.path.join(datasets_dir, 'defects')

    orig_defects1_dir = r'C:\Users\Alex\Downloads\Номера+дефекты\Номера+дефекты\defects1\defects1'
    orig_defects2_dir = r'C:\Users\Alex\Downloads\Номера+дефекты\Номера+дефекты\defects2\defects2'

    splits = ['train', 'valid', 'test']

    # Get dirs with images
    defects1_images_dirs = [os.path.join(defects1_dir, split, 'images') for split in splits]
    defects2_images_dirs = [os.path.join(defects2_dir, split, 'images') for split in splits]

    # Get set of all image filenames for each input dataset
    defects1_file_names = get_file_names(defects1_images_dirs)
    defects2_file_names = get_file_names(defects2_images_dirs)

    # Create output dirs
    defects_rename_dir = os.path.join(datasets_dir, 'defects_rename')
    for split in splits:
        for data_type in ['labels', 'images']:
            os.makedirs(os.path.join(defects_rename_dir, split, data_type), exist_ok=True)

    for split in splits:
        split_file_names = get_file_names([os.path.join(defects_dir, split, 'images')])
        print('split_file_names', len(split_file_names))

        for fn in split_file_names:
            if fn in defects1_file_names:
                postfix = '_1'
                src_img_dir = orig_defects1_dir
                src_lbl_dir = defects1_dir
            elif fn in defects2_file_names:
                postfix = '_2'
                src_img_dir = orig_defects2_dir
                src_lbl_dir = defects2_dir
            else:
                raise ValueError

            name, ext = os.path.splitext(fn)
            new_name = name.split('_')[0] + postfix
            # print(new_name)
            shutil.copy(os.path.join(src_img_dir, fn.split('_')[0] + '.png'),
                        os.path.join(defects_rename_dir, split, 'images', new_name + '.png'))
            shutil.copy(os.path.join(src_lbl_dir, split, 'labels', name + '.txt'),
                        os.path.join(defects_rename_dir, split, 'labels', new_name + '.txt'))

    # Add missed 329.png
    annotation_2 = r'C:\Users\Alex\Downloads\defects2.json'
    extr = Extractor()
    inst = Installer()
    data = extr.extract(annotation_2)
    inst.write_label_file(os.path.join(defects_rename_dir, 'train', 'labels'),
                          '329_2.txt',
                          data['annotations']['329.png'])
    inst.copy_image_file(os.path.join(orig_defects2_dir, '329.png'),
                         os.path.join(defects_rename_dir, 'train', 'images', '329_2.png'))

    # Some checks
    for split in splits:
        for data_type in ['images', 'labels']:
            num_of_files = len(os.listdir(os.path.join(defects_dir, split, data_type)))
            new_num_of_files = len(os.listdir(os.path.join(defects_rename_dir, split, data_type)))
            print(split, data_type, num_of_files, new_num_of_files)
            assert num_of_files == new_num_of_files

            if split == 'train':
                new_file_names = os.listdir(os.path.join(defects_rename_dir, split, data_type))
                new_names = [fn.split('_')[0] for fn in new_file_names]
                new_names.sort()

                file_names = os.listdir(os.path.join(defects_dir, split, data_type))
                names = [fn.split('_')[0] for fn in file_names]
                names.sort()

                print(names)
                print(new_names)


