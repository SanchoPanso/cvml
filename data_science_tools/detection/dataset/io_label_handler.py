class IOLabelHandler:
    def __init__(self):
        pass

    def read_labels(self, path: str) -> list:
        with open(path) as f:
            rows = f.read().split('\n')
            lines = []
            for row in rows:
                if row == '':
                    continue
                lines.append(list(map(float, row.split(' '))))
        return lines

    def write_labels(self, path: str, lines: list):
        with open(path, 'w') as f:
            for line in lines:
                f.write(' '.join(list(map(str, line))) + '\n')


