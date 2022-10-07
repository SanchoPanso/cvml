import os
import glob
import cv2
import numpy as np
from cvml.core.detector import Detector


def save_txt(img_size: tuple, detector_output: np.array, save_path: str):
  with open(save_path, 'w') as f:
    for i in range(detector_output.shape[0]):
      x, y, w, h, cls_conf, cls_id = detector_output[i]
      y /= img_size[0]
      h /= img_size[0]
      x /= img_size[1]
      w /= img_size[1]

      line = ' '.join(list(map(str, [int(cls_id), x + w/2, y + h/2, w, h, cls_conf])))
      f.write(line + '\n')



def detect_and_save(img_dir: str, save_dir: str, predictor: Detector, conf: float = 0.1):
  img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
  for img_file in img_files:
    img_name = os.path.split(img_file)[-1].replace(".jpg", "")
    print(img_name)
    img = cv2.imread(img_file)
    pred = predictor(img[:, :, ::-1], conf=conf)

    os.makedirs(save_dir, exist_ok=True)
    save_txt(img.shape, pred, os.path.join(save_dir, img_name + '.txt'))

