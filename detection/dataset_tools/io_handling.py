def read_yolo_labels(path: str) -> list:
    with open(path) as f:
        rows = f.read().split('\n')
        lines = []
        for row in rows:
            if row == '':
                continue
            lines.append(list(map(float, row.split(' '))))
    return lines


def write_yolo_labels(path: str, lines: list):
    with open(path, 'w') as f:
        for line in lines:
            f.write(' '.join(list(map(str, line))) + '\n')

