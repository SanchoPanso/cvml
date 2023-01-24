import os

from extractor import Extractor
from installer import Installer


def change_lines(lines: list, changes: dict) -> list:
    for line in lines:
        line[0] = changes[line[0]]
    return lines


download_dir = r'C:\Users\Alex\Downloads'
annotation_1 = os.path.join(download_dir, 'annotations_csv1_comet_1_new.json')
annotation_2 = os.path.join(download_dir, 'annotations_csv1_comet_2_new.json')
dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets'
input_dataset = os.path.join(dataset_dir, 'defects_rename')
output_dataset = os.path.join(dataset_dir, 'defects_joint')

xtr = Extractor()
data_1 = xtr.extract(annotation_1)
data_2 = xtr.extract(annotation_2)

data_1['annotations'] = {
    key.split('.')[0] + '_1.png': data_1['annotations'][key] for key in data_1['annotations'].keys()
}
data_2['annotations'] = {
    key.split('.')[0] + '_2.png': data_2['annotations'][key] for key in data_2['annotations'].keys()
}

data = data_1
data['annotations'].update(data_2['annotations'])
print(sorted(list(data['annotations'].keys())))

inst = Installer()
splits = ['train', 'valid', 'test']
for split in splits:
    print(split)
    for data_type in ['images', 'labels']:
        os.makedirs(os.path.join(output_dataset, split, data_type), exist_ok=True)
    images = os.listdir(os.path.join(input_dataset, split, 'images'))
    for image in images:
        print(image)
        name, ext = os.path.splitext(image)
        lines = data['annotations'][image]
        lines = change_lines(lines, {0: 0, 1: 0, 2: 1})
        inst.write_label_file(os.path.join(output_dataset, split, 'labels'),
                              name + '.txt',
                              lines)
        inst.copy_image_file(os.path.join(input_dataset, split, 'images', image),
                             os.path.join(output_dataset, split, 'images', image))


