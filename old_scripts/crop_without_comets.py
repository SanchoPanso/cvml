import os
import random
import cv2
import numpy as np
import shutil

CROP_SIZE = (128, 128)


def save_crop_images_by_labels(images_dir, labels_dir, save_dir, crop_amount):
    os.makedirs(save_dir, exist_ok=True)
    img_files = os.listdir(images_dir)
    lbl_files = os.listdir(labels_dir)

    img_files.sort()
    lbl_files.sort()

    cnt = 1
    for i in range(len(img_files)):
        img_file = img_files[i]
        # lbl_file = lbl_files[i]
        print(img_file)

        # with open(os.path.join(labels_dir, lbl_file)) as f:
        #     text = f.read().strip()
        #
        # if text != '':
        #     print(img_file, 'is not empty')
        #     continue

        img = cv2.imread(os.path.join(images_dir, img_file))
        crops_number = random.randint(5, 10)

        for j in range(crops_number):
            x = random.randint(0, img.shape[1] - CROP_SIZE[1])
            y = random.randint(0, img.shape[0] - CROP_SIZE[0])
            w = CROP_SIZE[1]
            h = CROP_SIZE[0]
            crop = img[y:y + h, x:x + w]

            cv2.imwrite(os.path.join(save_dir, f'wc_{cnt}.jpg'), crop)
            np.savez(os.path.join(save_dir, f'wc_{cnt}.npz'), arr_0=np.zeros(CROP_SIZE, dtype='uint8'))
            cnt += 1

            if cnt >= crop_amount:
                return


def save_crop_with_splitting(images_dir, dataset_dir, splitting):
    img_files = os.listdir(images_dir)
    img_files.sort()

    cnt = 0
    for key in splitting.keys():
        os.makedirs(os.path.join(dataset_dir, key), exist_ok=True)
        for i in range(splitting[key]):
            shutil.copy(os.path.join(images_dir, img_files[cnt]),
                        os.path.join(dataset_dir, key, f'fp_{cnt}_' + img_files[cnt].split('.')[0] + '.jpg'))

            img = cv2.imread(os.path.join(images_dir, img_files[cnt]))
            np.savez(os.path.join(dataset_dir, key, f'fp_{cnt}_' + img_files[cnt].split('.')[0] + '.npz'),
                     arr_0=np.zeros(img.shape[0:2], dtype='uint8'))
            cnt += 1
    return


datasets_dir = r'E:\PythonProjects\AnnotationConverter\datasets'
defects_dir = os.path.join(datasets_dir, 'defects_21082022_tubes')
fp_dir = r'C:\Users\Alex\Downloads\fp-20220825T085450Z-001\fp'
save_dir = os.path.join(datasets_dir, 'segment_none_comets_26082022')

defects_amount_train = 80

# save_crop_images_by_labels(os.path.join(defects_dir),
#                            os.path.join(defects_dir),
#                            os.path.join(save_dir, 'train'),
#                            80)
# save_crop_images_by_labels(os.path.join(defects_dir),
#                            os.path.join(defects_dir),
#                            os.path.join(save_dir, 'valid'),
#                            20)
save_crop_with_splitting(fp_dir,
                         save_dir,
                         splitting={'train': 20, 'valid': 6})









