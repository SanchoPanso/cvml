import os
import cv2
import glob


def get_labels(path: str) -> list:
    with open(path) as f:
        rows = f.read().split('\n')
        lines = []
        for row in rows:
            if row == '':
                continue
            lines.append(list(map(float, row.split(' '))))
    return lines


def get_file_number(file_name):
    name = os.path.split(file_name)[-1]
    number = int(name.split('.')[0].split('_')[-1])
    return number


def get_obj_fname(class_name: str, save_dir: str) -> str:
    files = glob.glob(os.path.join(save_dir, f'{class_name}_*.jpg'))

    if len(files) == 0:
        return f'{class_name}_1.jpg'

    last_number = max([get_file_number(file) for file in files])
    return f'{class_name}_{last_number + 1}.jpg'


datasets_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets')
defects_dir = os.path.join(datasets_dir, 'defects_05082022')
classes_images_dir = os.path.join(datasets_dir, 'defects_05082022_crops')
class_names = ['comet', 'joint']

# make dirs for crops
for cls in class_names:
    os.makedirs(os.path.join(classes_images_dir, cls), exist_ok=True)

for split in ['test', 'train', 'valid']:
    img_files = os.listdir(os.path.join(defects_dir, split, 'images'))
    img_files.sort()

    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        labels = get_labels(os.path.join(defects_dir, split, 'labels', img_name + '.txt'))
        img = cv2.imread(os.path.join(defects_dir, split, 'images', img_file))
        for label in labels:
            cls_id, xc, yc, w, h = label
            cls_id = int(cls_id)
            xc = int(xc * img.shape[1])
            yc = int(yc * img.shape[0])
            w = int(w * img.shape[1])
            h = int(h * img.shape[0])

            obj_fname = get_obj_fname(class_names[cls_id],
                                      os.path.join(classes_images_dir, class_names[cls_id]))
            img_crop_obj = img[yc - h//2: yc + h//2, xc - w//2: xc + w//2, :]
            cv2.imwrite(os.path.join(classes_images_dir, class_names[cls_id], obj_fname),
                        img_crop_obj)

            print(split, img_file, obj_fname)




