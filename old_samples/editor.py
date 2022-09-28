import os


class Editor:
    def __init__(self):
        pass

    def change_classes(self, data: dict, change_dict: dict, new_classes: list) -> dict:
        data['classes'] = new_classes
        for key in data['annotations'].keys():
            lines = data['annotations'][key]
            new_lines = []

            for line in lines:
                new_line = line
                if change_dict[line[0]] is not None:
                    new_line[0] = change_dict[line[0]]
                    new_lines.append(new_line)
            data['annotations'][key] = new_lines
        return data

    def change_image_name(self, data: dict, update_func) -> dict:
        images = list(data['annotations'].keys())
        for image in images:
            name, ext = os.path.splitext(image)
            new_image = update_func(name) + ext
            data['annotations'][new_image] = data['annotations'].pop(image)
        return data
