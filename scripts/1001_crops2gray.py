import cv2
import os

src_dir = '/home/student2/datasets/crops/0712_comet_crops/comet'
dst_dir = '/home/student2/datasets/crops/0712_gray_comet_crops/comet'

os.makedirs(dst_dir, exist_ok=True)

files = os.listdir(src_dir)
for f in files:
    img = cv2.imread(os.path.join(src_dir, f))
    _, _, gray = cv2.split(img)
    cv2.imwrite(os.path.join(dst_dir, f), gray)
    print(f)


