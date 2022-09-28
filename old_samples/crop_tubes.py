import os
import sys
import numpy as np
import cv2

from detection.inference_tools.yolov5_detector import Yolov5Detector

if __name__ == '__main__':
    weights_path = r'E:\PythonProjects\AnnotationConverter\weights\best_5.pt'
    images_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_21082022\train\images'
    labels_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_21082022\train\labels'
    save_dir = r'E:\PythonProjects\AnnotationConverter\datasets\defects_21082022_tubes'

    detector = Yolov5Detector(weights_path)

    img_files = os.listdir(images_dir)
    lbl_files = os.listdir(labels_dir)
    img_files.sort()
    lbl_files.sort()

    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for i, img_file in enumerate(img_files):
        print(img_file)

        with open(os.path.join(labels_dir, lbl_files[i])) as f:
            text = f.read().strip()
        if text != '':
            print('label is not empty')
            continue

        img = cv2.imread(os.path.join(images_dir, img_file))
        img_rgb = img[:, :, ::-1]
        pred = detector(img_rgb, conf=0.25)

        if pred.shape[0] == 0:
            print('pred is empty')
            continue

        # squares = [pred[j][2] * pred[j][3] for j in range(pred.shape[0])]
        # max_idx = np.array(squares).argmax()
        x, y, w, h = map(int, pred[0][0:4])
        crop_img = img[y:y + h, x:x + w]

        cv2.imwrite(os.path.join(save_dir, img_file), crop_img)
        cnt += 1

        if cnt > 100:
            break

