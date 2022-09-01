import os

yolov5_dataset_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets', 'defects_rename')

for split in ['train', 'test', 'valid']:
    labels_files = os.listdir(os.path.join(yolov5_dataset_dir, split, 'labels'))
    for file in labels_files:
        with open(os.path.join(yolov5_dataset_dir, split, 'labels', file), 'r') as f:
            rows = f.read().split('\n')

        for i in range(len(rows)):
            if rows[i].strip() != '':
                rows[i] = '0' + rows[i][1:]

        with open(os.path.join(yolov5_dataset_dir, split, 'labels', file), 'w') as f:
            f.write('\n'.join(rows))


