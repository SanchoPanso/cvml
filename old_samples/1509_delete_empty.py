import os
from detection.dataset_tools.io_handling import read_yolo_labels


dataset_dir = 'F:\\datasets\\tmk_yolov5_15092022'
cnt = 0

if __name__ == '__main__':
    splits = ['test', 'train', 'valid']

    for split in splits:
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        images_dir = os.path.join(dataset_dir, split, 'images')
        txt_files = os.listdir(labels_dir)

        for txt_file in txt_files:
            labels = read_yolo_labels(os.path.join(labels_dir, txt_file))
            if len(labels) == 0:
                name, ext = os.path.splitext(txt_file)
                os.remove(os.path.join(labels_dir, txt_file))
                os.remove(os.path.join(images_dir, name + '.jpg'))
                print(name)
                cnt += 1
                print(cnt)




