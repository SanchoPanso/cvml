import os


def change_classes_in_lines(lines: list, change_dict: dict) -> list:
    new_lines = []
    for line in lines:
        new_line = line
        if change_dict[int(line[0])] is not None:
            new_line[0] = change_dict[int(line[0])]
            new_lines.append(new_line)
    return new_lines


def read_labels(path: str) -> list:
    with open(path) as f:
        rows = f.read().split('\n')
        lines = []
        for row in rows:
            if row == '':
                continue
            lines.append(list(map(float, row.split(' '))))
    return lines


def write_labels(path: str, lines: list):
    with open(path, 'w') as f:
        for line in lines:
            f.write(' '.join(list(map(str, line))) + '\n')


if __name__ == '__main__':
    orig_dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_05082022_augmented2'
    dataset_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_21082022'
    changes = {0: 0, 1: None}
    for split in ['test', 'train', 'valid']:
        files = os.listdir(os.path.join(dataset_dir, split, 'labels'))
        for file in files:
            print(split, file)
            lines = read_labels(os.path.join(orig_dataset_dir, split, 'labels', file))
            new_lines = change_classes_in_lines(lines, changes)
            write_labels(os.path.join(dataset_dir, split, 'labels', file), new_lines)
            print(lines)
            print(new_lines)


